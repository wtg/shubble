import './styles/Feedback.css';

export default function Feedback() {
  return (
    <div className='flex-feedback'>
      <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 16 16">
        <g fill="currentColor">
          <path d="m4.5 1l-.5.5v1.527a4.6 4.6 0 0 1 1 0V2h9v5h-1.707L11 8.293V7H8.973a4.6 4.6 0 0 1 0 1H10v1.5l.854.354L12.707 8H14.5l.5-.5v-6l-.5-.5z" />
          <path fillRule="evenodd" d="M6.417 10.429a3.5 3.5 0 1 0-3.834 0A4.5 4.5 0 0 0 0 14.5v.5h1v-.5a3.502 3.502 0 0 1 7 0v.5h1v-.5a4.5 4.5 0 0 0-2.583-4.071M4.5 10a2.5 2.5 0 1 1 0-5a2.5 2.5 0 0 1 0 5" clipRule="evenodd" />
        </g>
      </svg>
      <p className='feedback-p'>
        Have feedback? Please fill out <a href='https://forms.office.com/r/VTC45CRPV3'>our feedback form</a>!
      </p>
    </div>
  );
}
