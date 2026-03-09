export interface Announcement {
    id: string;
    message: string;
    type: 'info' | 'warning' | 'error';
    active: boolean;
    expires_at?: string;
    created_at: string;
    upvotes: number;
    downvotes: number;
}

export interface AnnouncementsData {
    announcements: Announcement[];
}
