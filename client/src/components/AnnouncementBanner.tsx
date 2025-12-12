import '../styles/AnnouncementBanner.css';
import announcementsData from '../data/announcements.json';
import type { Announcement, AnnouncementsData } from '../ts/types/announcement';
import type { ReactNode } from 'react';

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
function getActiveAnnouncements(data: AnnouncementsData): Announcement[] {
    const now = new Date();

    return data.announcements.filter((announcement) => {
        // Skip inactive announcements
        if (!announcement.active) return false;

        // Skip expired announcements
        if (announcement.expiresAt) {
            const expirationDate = new Date(announcement.expiresAt);
            if (expirationDate < now) return false;
        }

        return true;
    });
}

/**
 * Displays announcements loaded from the JSON file.
 * Only shows active, non-expired announcements.
 */
export default function AnnouncementBanner() {
    const activeAnnouncements = getActiveAnnouncements(announcementsData as AnnouncementsData);

    if (activeAnnouncements.length === 0) {
        return null;
    }

    return (
        <>
            {activeAnnouncements.map((announcement) => (
                <Banner
                    key={announcement.id}
                    message={announcement.message}
                    type={announcement.type}
                />
            ))}
        </>
    );
}
