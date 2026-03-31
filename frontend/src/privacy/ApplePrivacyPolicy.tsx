import './ApplePrivacyPolicy.css';

function ApplePrivacyPolicy() {
  return (
    <div className="privacy-policy">
      <h1>Privacy Policy</h1>
      <p className="privacy-effective">Effective Date: March 10, 2026</p>

      <section>
        <h2>Introduction</h2>
        <p>
          Shubble (&quot;we&quot;, &quot;our&quot;, or &quot;us&quot;) provides a real-time shuttle tracking
          application for Rensselaer Polytechnic Institute (RPI). This Privacy
          Policy explains how we collect, use, and protect information when you
          use our mobile application and website.
        </p>
      </section>

      <section>
        <h2>Information We Collect</h2>
        <h3>Information We Do Not Collect</h3>
        <ul>
          <li>We do not collect any personal information from users.</li>
          <li>We do not require account creation or login.</li>
          <li>We do not collect names, email addresses, or phone numbers.</li>
          <li>We do not track or store your location.</li>
        </ul>

        <h3>Information Automatically Collected</h3>
        <ul>
          <li>
            <strong>Anonymous Analytics:</strong> We use Umami, a
            privacy-focused analytics tool, to collect anonymous usage
            statistics such as page views and general device type. This data
            cannot be used to identify individual users.
          </li>
        </ul>
      </section>

      <section>
        <h2>Shuttle Location Data</h2>
        <p>
          Our app displays real-time shuttle locations provided by the Samsara
          fleet management API. This data pertains to shuttle vehicles only, 
          not to app users. User data is enabled only for the purposes of local app experience, and is not transmitted to any server.
        </p>
      </section>

      <section>
        <h2>Data Sharing</h2>
        <p>
          We do not sell, trade, or share any user data with third parties. The
          anonymous analytics data collected by Umami is used solely to improve
          the app experience.
        </p>
      </section>

      <section>
        <h2>Data Storage &amp; Security</h2>
        <p>
          Since we do not collect personal user data, there is no personal
          information at risk. Shuttle GPS data is stored temporarily in our
          servers for real-time display and prediction purposes only.
        </p>
      </section>

      <section>
        <h2>Children&apos;s Privacy</h2>
        <p>
          Our app does not collect personal information from anyone, including
          children under 13 years of age.
        </p>
      </section>

      <section>
        <h2>Changes to This Policy</h2>
        <p>
          We may update this Privacy Policy from time to time. Any changes will
          be posted on this page with an updated effective date.
        </p>
      </section>

      <section>
        <h2>Contact Us</h2>
        <p>
          If you have any questions about this Privacy Policy, please contact us
          at: <a href="mailto:krishj4@rpi.edu">krishj4@rpi.edu</a>
        </p>
      </section>
    </div>
  );
}

export default ApplePrivacyPolicy;
