export default function WarningBanner({ bannerText, bannerLink }) {
    return (
        <div className="banner">
            {bannerLink && (
                <p>
                    This is a staging domain. Please visit our official website{" "}
                    <a href={bannerLink}>here</a>!
                </p>
            )}
        </div>
    );
    }