import '../styles/About.css';
import {
  useState,
  useEffect,
} from 'react';
import {
  Link
} from 'react-router';

export default function About() {

  const [wordIndex, setWordIndex] = useState(0);
  const words = ['Reliable', 'Predictable', 'Accountable'];

  // Rotate words every 2 seconds
  useEffect(() => {
    setInterval(() => {
      setWordIndex((prevIndex) => {
        return (prevIndex + 1) % words.length;
      });
    }, 2000);
  }, []);

  return (
    <div id ="grad">
      <div className="about-wrapper">
        <div className="about-left">
          <p className="about-small-title">Making Shuttles</p>
          <h1 className="about-main-header"><span className='word-rotator'>{words[wordIndex]}</span></h1>
          <div className="about-description">
            <h1>Track RPI shuttles with live location and view schedules with Shubble.<br /><br /><br /></h1>
            <p>
              Shubble is an open source project under the Rensselaer Center for Open Source (RCOS).<br />
              Have an idea to improve it? Contributions are welcome!<br /> 
              Visit our <a href='https://github.com/wtg/shubble' target='_blank'>Github Repository</a> to learn more.<br />
              Interested in Shubble's data? Take a look at our
              <Link to='/data'>
                <span className = 'link1'>data page</span>
              </Link>.
            </p>
          </div>
        </div>

        <div className="about-right">
          <img src="/RPIStudentUnionv2.png" alt="RPI Student Union" className="about-image" />
        </div>
      </div>
    </div>
  )
}
