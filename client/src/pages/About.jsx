import '../styles/About.css';
import { } from 'react';
import {
  Link
} from 'react-router';
import TextAnimation from '../components/TextAnimation';

export default function About() {
  const words = ['Reliable', 'Predictable', 'Accountable'];

  //calculates on mount, didn't feel it was necessary to use a hook, not technically responsive, but doesn't need to be
  const screenWidth = window.innerWidth;
  const isMobile = screenWidth < 1025;

  return (
    <div className='about'>
      <section className='hero'>
        <div className='hero-content'>
          <h1 className='hero-title'>Making Shuttles</h1>
          <h1 className='hero-subtitle'>
            <TextAnimation 
              words={words}
              height={isMobile ? 60 : 70}
              fontSize={isMobile ? 50 : 60}
            />
          </h1>
          <p className='hero-description'>
            Track RPI campus shuttles in real-time with Shubble - a reliable shuttle tracking system built for students, by students.
          </p>
          
          <div className='hero-actions'>
            <Link to='/schedule' className='btn btn-primary'>
              Explore Schedules
            </Link>
            <Link to='/' className='btn btn-secondary'>
              Live Location
            </Link>
          </div>
        </div>
      </section>


      <section className='about-section'>
        <div className='container'>
          <h2 className='section-title'>About Shubble</h2>
          <div className='about-content'>
            <div className='about-text'>
              <p>
                Shubble is the latest shuttle tracker, built using modern web technologies including MapKit JS, React, and Flask. 
                Our mission is to make campus transportation more reliable and predictable for students.
              </p>
              <p>
                As an open source project under the Rensselaer Center for Open Source (RCOS), Shubble represents the power 
                of collaborative development and student innovation.
              </p>
              <p>
                Have an idea to improve it? Contributions are welcome! Visit our 
                <a href='https://github.com/wtg/shubble' target='_blank' rel='noopener noreferrer'> GitHub Repository</a> to learn more.
              </p>
            </div>
            <div className='about-actions'>
              <Link to='/data' className='btn btn-outline'>
                Explore Data
              </Link>
            </div>
          </div>
        </div>
      </section>

    </div>
  )
}
