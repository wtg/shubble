export interface Announcement {
    id: string;
    message: string;
    type: 'info' | 'warning' | 'error';
    active: boolean;
    expiresAt?: string;
    createdAt: string;
}

export interface AnnouncementsData {
    announcements: Announcement[];
}
