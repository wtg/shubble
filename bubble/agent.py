"""Gemini-powered announcement management for Bubble."""
import json
import logging
from datetime import datetime, timedelta, timezone
from backend.config import settings as backend_settings
from typing import Literal, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, TypeAdapter, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from backend.models import Announcement, SuggestedAnnouncement

logger = logging.getLogger(__name__)

_BUBBLE_IDS_KEY = "bubble:announcement_ids"
_MAX_EXPIRES_HOURS = 72.0

_SYSTEM_PROMPT = """You are Bubble, an AI assistant for the RPI Shuttle Tracker (Shubble).
Your job is to manage announcements for shuttle riders at Rensselaer Polytechnic Institute (RPI)
in Troy, New York.

You will be given the following inputs, listed from HIGHEST to LOWEST priority:

1. Live shuttle data from the tracker API — this is real-time ground truth. It always
   overrides claims made by external sources. Apply these rules without exception:
   - If live data shows one or more active shuttles: shuttles ARE running. Do NOT announce
     any service suspension, regardless of what external sources say.
   - If live data shows no active shuttles AND an external source confirms a closure for
     today: announce the suspension.
   - If live data is unavailable: fall back to external sources.

2. Your currently active announcements (which you may keep, modify, or delete)

3. Manually-created announcements (read-only context — do not include these in your output)

4. Admin-submitted suggestions — treat these as strong hints. Create an announcement
   reflecting their concern unless it clearly doesn't meet the criteria above.
   HOWEVER: if you previously created an announcement based on a suggestion and users
   significantly downvoted it, the suggestion is likely inaccurate or misleading.
   In that case, do NOT recreate a similar announcement unless live data or a
   reliable external source independently confirms the concern.

5. Current data from external sources — use only when live data is unavailable or does
   not address the situation (e.g. upcoming schedule changes, weather, road closures).

Generate an announcement whenever any of the following applies TODAY or in the next 48 hours:
- Shuttles are not operating due to a break, holiday, or school closure
- A specific date range of no service includes today's date
- Shuttles are running on a reduced or modified schedule
- Weather conditions may delay or cancel shuttle service
- Road closures, construction, or campus events are affecting routes

IMPORTANT: If the source lists service suspension dates (e.g. "no service Spring Break 2-28 thru 3-7")
and today's date falls within that range, BUT live data shows active shuttles, do NOT announce
a suspension — the live tracker is the authoritative source.

Output rules:
- Return the COMPLETE desired state of your announcements as a JSON array
- To KEEP or MODIFY an existing announcement: include it with its original "id" field
- To CREATE a new announcement: include it without an "id" field
- To DELETE an existing announcement: simply omit it from the array
- Do NOT include manually-created announcements in your output
- ONE announcement per distinct subject (e.g. service suspension and weather are separate announcements)
- Review the 👍/👎 counts on your announcements. If downvotes are significant (use judgment: at least 3 net downvotes for a minor notice, more for a major service alert), users likely found it inaccurate or unhelpful — remove or substantially revise it based on current data
- Check your recent decision history carefully. If you removed or substantially revised an announcement due to downvotes in a previous run, do NOT recreate a similar announcement in this run unless new evidence has emerged (e.g. live shuttle data has changed, or an external source now confirms the situation). Recreating a recently rejected announcement is not acceptable and will result in the same user rejection.
- Keep each message to 1-2 sentences; markdown links are supported
- Use "warning" for service reductions, "error" for full suspensions, "info" for general notices
- Set expires_in_hours to cover until the end of the affected period (max 72)
- Return [] only if there are no relevant announcements needed

Return ONLY a JSON array — no other text:
[
  {"id": 5, "message": "...", "type": "warning", "expires_in_hours": 24},
  {"message": "...", "type": "info", "expires_in_hours": 12}
]"""


class GeneratedAnnouncement(BaseModel):
    """One announcement in Gemini's output.

    id present  → keep or modify that existing Bubble-managed announcement
    id absent   → create a new announcement
    Bubble announcements omitted from the output are deleted.
    """

    id: Optional[int] = None
    message: str
    type: Literal["info", "warning", "error"] = "info"
    expires_in_hours: float = Field(default=24.0, gt=0)

    @field_validator("expires_in_hours")
    @classmethod
    def clamp_expires(cls, v: float) -> float:
        return min(v, _MAX_EXPIRES_HOURS)


_adapter = TypeAdapter(list[GeneratedAnnouncement])


async def get_current_announcements(
    session_factory: async_sessionmaker,
    redis_client,
) -> tuple[list[Announcement], list[Announcement], list[SuggestedAnnouncement]]:
    """Return (bubble_managed, manual, suggestions)."""
    now = datetime.now(timezone.utc)

    ids_raw = await redis_client.get(_BUBBLE_IDS_KEY)
    bubble_ids = set(json.loads(ids_raw)) if ids_raw else set()

    async with session_factory() as session:
        ann_result = await session.execute(
            select(Announcement).where(
                Announcement.active == True,
                Announcement.expires_at >= now,
            )
        )
        all_announcements = ann_result.scalars().all()

        sug_result = await session.execute(
            select(SuggestedAnnouncement).order_by(SuggestedAnnouncement.created_at.desc())
        )
        suggestions = sug_result.scalars().all()

    bubble = [a for a in all_announcements if a.id in bubble_ids]
    manual = [a for a in all_announcements if a.id not in bubble_ids]
    return bubble, manual, suggestions


async def generate_announcements(
    source_data: dict[str, str],
    api_key: str,
    model: str,
    bubble_announcements: list[Announcement],
    manual_announcements: list[Announcement],
    suggestions: list[SuggestedAnnouncement],
    live_location_summary: str | None = None,
    past_exchanges: list[tuple[str, str]] | None = None,
) -> tuple[list[GeneratedAnnouncement], str, str]:
    """Call Gemini with the current announcement state and source data.

    Returns (announcements, user_content, raw_response_text).
    user_content and raw_response_text are provided so the caller can persist
    the exchange to memory.  On generation failure, both strings are empty.
    """
    client = genai.Client(api_key=api_key)

    now = datetime.now(backend_settings.CAMPUS_TZ)
    today = now.strftime("%A, %B %-d, %Y")
    parts = [f"Today's date is {today} (campus timezone: {backend_settings.CAMPUS_TZ}).\n"]

    if bubble_announcements:
        parts.append("Your currently active announcements (keep, modify, or delete as needed):")
        for ann in bubble_announcements:
            hours_left = (ann.expires_at - now).total_seconds() / 3600
            parts.append(f'  - ID {ann.id}: [{ann.type}] "{ann.message}" (expires in {hours_left:.0f}h, 👍 {ann.upvotes} 👎 {ann.downvotes})')
        parts.append("")
    else:
        parts.append("You have no currently active announcements.\n")

    if manual_announcements:
        parts.append("Manually-created announcements (read-only context, do NOT include in your output):")
        for ann in manual_announcements:
            parts.append(f'  - [{ann.type}] "{ann.message}"')
        parts.append("")

    if suggestions:
        parts.append("Admin-submitted suggestions (treat as strong hints):")
        for s in suggestions:
            parts.append(f'  - "{s.suggestion}"')
        parts.append("")

    if live_location_summary is not None:
        parts.append(
            f"Live shuttle data (HIGHEST PRIORITY — overrides external sources):\n"
            f"{live_location_summary}\n"
            f"RULE: If any shuttles are active above, do NOT announce a service suspension.\n"
        )
    else:
        parts.append("Live shuttle data: unavailable — rely on external sources alone.\n")

    if past_exchanges:
        parts.append("Your recent decision history (oldest first — for context only):")
        for _, ai_response in past_exchanges:
            parts.append(f"  {ai_response}")
        parts.append("")

    if source_data:
        parts.append("Current data from external sources:\n")
        for url, content in source_data.items():
            parts.append(f"**Source:** {url}\n```\n{content}\n```\n")
    else:
        parts.append("No external source data is available.\n")

    parts.append(
        "Return the complete desired state of your announcements as a JSON array. "
        "Include IDs for announcements you want to keep or modify, omit IDs for new ones, "
        "and simply exclude any announcement you want to delete."
    )

    user_content = "\n".join(parts)

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=[_SYSTEM_PROMPT, user_content],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        logger.debug("Gemini raw response: %s", response.text)
        announcements = _adapter.validate_json(response.text)
        logger.info("Gemini returned %d announcement(s)", len(announcements))
        return announcements, user_content, response.text
    except Exception as e:
        logger.error("Failed to parse Gemini response: %s", e)
        return [], user_content, ""


def _retire(ann: Announcement, now: datetime) -> None:
    """Soft-expire an announcement, preserving it for history."""
    ann.active = False
    ann.expires_at = now


async def update_bubble_announcements(
    session_factory: async_sessionmaker,
    redis_client,
    updates: list[GeneratedAnnouncement],
    bubble_announcements: list[Announcement],
) -> None:
    """Apply Gemini's desired announcement state to the database.

    History is preserved: rows are never deleted.

    - updates with id in bubble_announcements, wording unchanged → extend expires_at only
    - updates with id in bubble_announcements, wording changed  → retire old, insert new
    - updates with id NOT in bubble_announcements               → skip (protect manual rows)
    - updates without id                                        → insert new row
    - bubble_announcements not mentioned in updates             → retire (soft-expire)
    """
    now = datetime.now(timezone.utc)
    bubble_id_map = {ann.id: ann for ann in bubble_announcements}

    mentioned_ids = {u.id for u in updates if u.id is not None}
    ids_to_retire = set(bubble_id_map) - mentioned_ids

    new_ids: list[int] = []
    n_retired = 0
    n_replaced = 0
    n_extended = 0
    n_created = 0

    async with session_factory() as session:
        # Retire announcements Gemini dropped
        for ann_id in ids_to_retire:
            ann = await session.get(Announcement, ann_id)
            if ann:
                _retire(ann, now)
                n_retired += 1
        if ids_to_retire:
            logger.debug("Retired %d Bubble announcement(s)", n_retired)

        for update in updates:
            if update.id is not None:
                if update.id not in bubble_id_map:
                    logger.warning(
                        "Gemini referenced non-Bubble announcement id=%d — skipping", update.id
                    )
                    continue

                old = await session.get(Announcement, update.id)
                if old is None:
                    continue

                wording_changed = old.message != update.message or old.type != update.type
                if wording_changed:
                    # Retire old version, create a new row to preserve history.
                    # Carry votes forward so user reactions aren't lost on rewording.
                    _retire(old, now)
                    new_ann = Announcement(
                        message=update.message,
                        type=update.type,
                        active=True,
                        expires_at=now + timedelta(hours=update.expires_in_hours),
                        upvotes=0,
                        downvotes=0,
                    )
                    session.add(new_ann)
                    await session.flush()
                    new_ids.append(new_ann.id)
                    n_replaced += 1
                else:
                    # Same wording — just extend the expiry
                    old.expires_at = now + timedelta(hours=update.expires_in_hours)
                    new_ids.append(old.id)
                    n_extended += 1
            else:
                # Brand-new announcement
                new_ann = Announcement(
                    message=update.message,
                    type=update.type,
                    active=True,
                    expires_at=now + timedelta(hours=update.expires_in_hours),
                    upvotes=0,
                    downvotes=0,
                )
                session.add(new_ann)
                await session.flush()
                new_ids.append(new_ann.id)
                n_created += 1

        await session.commit()

    await redis_client.set(_BUBBLE_IDS_KEY, json.dumps(new_ids))
    logger.info(
        "Bubble announcements updated: %d active (%d extended, %d replaced, %d new), %d retired",
        len(new_ids),
        n_extended,
        n_replaced,
        n_created,
        n_retired,
    )
