export default function WarningBanner({ bannerText, bannerLink }) {
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
        </div>
    );
    }