export default function WarningBanner({ bannerText, gitRev, bannerLink }) {
    return (
        <div className="banner">
            {bannerLink && (
                <p>
                    This is a staging domain. Please visit our official website{" "}
                    <a href={bannerLink}>here</a>!
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