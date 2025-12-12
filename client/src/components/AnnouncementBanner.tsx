import '../styles/AnnouncementBanner.css';
import announcementsData from '../data/announcements.json';
import type { Announcement, AnnouncementsData } from '../ts/types/announcement';

type BannerType = 'info' | 'warning' | 'error';

interface BannerProps {
    message: string;
    type: BannerType;
    showReload?: boolean;
}

/**
 * Standalone banner component for displaying messages with different severity levels.
 * Can be used by ErrorBoundary or any other component that needs to display a banner.
 */
export function Banner({ message, type, showReload = false }: BannerProps) {
    return (
        <div className={`announcement-banner ${type}`}>
            <p>{message}</p>
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
