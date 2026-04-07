import './styles/AnnouncementBanner.css';
import type { Announcement} from '../types/announcement';
import {
    useState,
    useEffect,
    useRef,
    type ReactNode
} from 'react';
import config from '../utils/config';

type BannerType = 'info' | 'warning' | 'error';

interface BannerProps {
    message: string;
    type: BannerType;
    showReload?: boolean;
}

/**
 * Parses a message string and converts markdown-style links [text](url) to React elements.
 * Only allows https:// URLs to prevent XSS attacks.
 * All other text is rendered as plain text (auto-escaped by React).
 */
function parseMessageWithLinks(message: string): ReactNode[] {
    const parts: ReactNode[] = [];
    // Regex to match [link text](url) pattern
    const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;

    let lastIndex = 0;
    let match;
    let keyIndex = 0;

    while ((match = linkRegex.exec(message)) !== null) {
        // Add text before the link
        if (match.index > lastIndex) {
            parts.push(message.slice(lastIndex, match.index));
        }

        const linkText = match[1];
        const url = match[2];

        // Security: Only allow https:// URLs
        if (url.startsWith('https://')) {
            parts.push(
                <a
                    key={keyIndex++}
                    href={url}
                    target="_blank"
                    rel="noopener noreferrer"
                >
                    {linkText}
                </a>
            );
        } else {
            // If URL is not https, render as plain text
            parts.push(`${linkText} (${url})`);
        }

        lastIndex = match.index + match[0].length;
    }

    // Add remaining text after last link
    if (lastIndex < message.length) {
        parts.push(message.slice(lastIndex));
    }

    return parts.length > 0 ? parts : [message];
}

/**
 * Standalone banner component for displaying messages with different severity levels.
 * Can be used by ErrorBoundary or any other component that needs to display a banner.
 * Supports markdown-style links: [link text](https://example.com)
 */
export function Banner({ message, type, showReload = false }: BannerProps) {
    return (
        <div className={`announcement-banner ${type}`}>
            <p>{parseMessageWithLinks(message)}</p>
            {showReload && (
                <button className="reload-btn" onClick={() => window.location.reload()}>
                    Reload Page
                </button>
            )}
        </div>
    );
}

/**
 * Filters announcements to only return active, non-expired ones.
 */
async function getActiveAnnouncements(): Promise<Announcement[]> {
    try{
        const response = await fetch(`${config.apiBaseUrl}/api/announcements`);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json() as Announcement[];
        return data
    } catch (error) {
        console.error('Error fetching announcements:', error);
        return [];
    }
}

async function postVote(
    id: string,
    direction: 'up' | 'down' | null,
    previousDirection: 'up' | 'down' | null,
): Promise<{ upvotes: number; downvotes: number } | null> {
    try {
        const response = await fetch(`${config.apiBaseUrl}/api/announcements/${id}/vote`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ direction, previous_direction: previousDirection }),
        });
        if (!response.ok) return null;
        return await response.json() as { upvotes: number; downvotes: number };
    } catch {
        return null;
    }
}

/**
 * Displays announcements one at a time, rotating every 4 seconds when there are multiple.
 * A three-dot menu on the right lists all announcements with thumbs up/down reactions.
 */
export default function AnnouncementBanner() {
    const [activeAnnouncements, setAnnouncements] = useState<Announcement[] | null>(null);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [menuOpen, setMenuOpen] = useState(false);
    const [reactions, setReactions] = useState<Record<string, 'up' | 'down' | null>>({});
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        getActiveAnnouncements().then(setAnnouncements);

        const interval = setInterval(() => {
            getActiveAnnouncements().then(setAnnouncements);
        }, 60_000);

        return () => clearInterval(interval);
    }, []);

    useEffect(() => {
        if (!activeAnnouncements || activeAnnouncements.length <= 1) return;

        const interval = setInterval(() => {
            setCurrentIndex(i => (i + 1) % activeAnnouncements.length);
        }, 4000);

        return () => clearInterval(interval);
    }, [activeAnnouncements]);

    useEffect(() => {
        if (!menuOpen) return;
        const onMouseDown = (e: MouseEvent) => {
            if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
                setMenuOpen(false);
            }
        };
        document.addEventListener('mousedown', onMouseDown);
        return () => document.removeEventListener('mousedown', onMouseDown);
    }, [menuOpen]);

    const toggleReaction = async (id: string, dir: 'up' | 'down') => {
        const previous = reactions[id] ?? null;
        const next = previous === dir ? null : dir;

        // Optimistically update the reaction state
        setReactions(prev => ({ ...prev, [id]: next }));

        const result = await postVote(id, next, previous);
        if (result) {
            // Update counts from the authoritative server response
            setAnnouncements(prev => prev?.map(ann =>
                ann.id === id ? { ...ann, upvotes: result.upvotes, downvotes: result.downvotes } : ann
            ) ?? null);
        }
    };

    const count = activeAnnouncements?.length ?? 0;
    if (count === 0) return null;

    const safeIndex = currentIndex % count;
    const announcement = activeAnnouncements![safeIndex];

    return (
        <div className="announcement-container" ref={containerRef}>
            <Banner key={safeIndex} message={announcement.message} type={announcement.type} />

            <button
                className={`announcement-menu-btn ${announcement.type}`}
                onClick={() => setMenuOpen(o => !o)}
                aria-label="Announcement options"
            >
                <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
                    <circle cx="2.5" cy="8" r="1.5" fill="currentColor" />
                    <circle cx="8"   cy="8" r="1.5" fill="currentColor" />
                    <circle cx="13.5" cy="8" r="1.5" fill="currentColor" />
                </svg>
            </button>

            {menuOpen && (
                <div className="announcement-menu">
                    {activeAnnouncements!.map(ann => (
                        <div key={ann.id} className="announcement-menu-row">
                            <span className="announcement-menu-row-text">
                                {ann.message}
                            </span>
                            <div className="announcement-menu-row-actions">
                                <button
                                    className={`reaction-btn${reactions[ann.id] === 'up' ? ' active' : ''}`}
                                    onClick={() => toggleReaction(ann.id, 'up')}
                                    aria-label="Thumbs up"
                                >👍 {ann.upvotes > 0 && <span className="reaction-count">{ann.upvotes}</span>}</button>
                                <button
                                    className={`reaction-btn${reactions[ann.id] === 'down' ? ' active' : ''}`}
                                    onClick={() => toggleReaction(ann.id, 'down')}
                                    aria-label="Thumbs down"
                                >👎 {ann.downvotes > 0 && <span className="reaction-count">{ann.downvotes}</span>}</button>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
