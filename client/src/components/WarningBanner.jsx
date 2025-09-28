export default function WarningBanner({ bannerText, gitRev, bannerLink }) {
    return (
        <div className="banner">
            <h1>{bannerText}</h1>
            {bannerLink && (
                <p>
                    <a href={bannerLink}>
                        Production Site
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