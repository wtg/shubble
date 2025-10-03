export default function WarningBanner({ bannerText, gitRev, bannerLink }) {
    return (
        <div className="banner">
            <p>{bannerText}</p>
            {bannerLink && (
                <p>
                    <a href={bannerLink}>
                    Official Website
                    </a>
                </p>
            )}
            {
                gitRev != 'unknown' ? (
                    <a href={`https://github.com/wtg/shubble/commit/${gitRev}`}>
                        {gitRev}
                    </a>
                ) :
                <p>Version unknown</p>
            }
        </div>
    );
}