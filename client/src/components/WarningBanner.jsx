export default function WarningBanner({ bannerText, gitRev, bannerLink }) {
    return (
        <div className="banner">
            <h1>{bannerText}</h1>
            <p>Git Revision: {gitRev}</p>
            {bannerLink && (
                <p>
                    <a href={bannerLink}>
                        Production Site
                    </a>
                </p>
            )}
        </div>
    );
    }