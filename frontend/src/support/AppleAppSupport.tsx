import './AppleAppSupport.css';

function AppleAppSupport() {
  return (
    <div className="app-support">
      <h1>App Support</h1>
      
      <section>
        <h2>Need Help?</h2>
        <p>
          Welcome to the Shubble App Support page. If you are experiencing issues with the iOS application, have questions, or want to provide feedback, please let us know using the form below. We will do our best to address your concerns promptly.
        </p>
        <p>
          For technical feature requests, bug reports, and to see what we are currently working on, please visit our <a href="https://github.com/wtg/Shuttle-Tracker-SwiftUI/issues" target="_blank" rel="noopener noreferrer">GitHub Issues page</a>.
        </p>
      </section>

      <section>
        <h2>Contact Support Form</h2>
        <div className="iframe-container">
          <iframe 
            src="https://forms.office.com/pages/responsepage.aspx?id=4qN6GTeoEEihSi_yiMjYTq8qULt9QhxLloyay1J-8pdUQjAzVFRDV0ZNNU9VTEdRMVVCVFhFNTNEMC4u&route=shorturl" 
            width="100%" 
            height="100%" 
            style={{ border: "none", maxWidth: "100%", maxHeight: "100%" }} 
            allowFullScreen 
            title="Support Form"
          ></iframe>
        </div>
      </section>
    </div>
  );
}

export default AppleAppSupport;
