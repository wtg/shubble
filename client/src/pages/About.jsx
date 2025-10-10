import '../styles/About.css';
import { } from 'react';
import {
  Link
} from 'react-router';
import TextAnimation from '../components/TextAnimation';

export default function About() {
  const words = ['Reliable', 'Predictable', 'Accountable'];

  return (
    <div className='about'>
      <h1>Making Shuttles</h1>
      <h1>
        <TextAnimation 
        words={words}
        height={40}
        fontSize={30}
         />
      </h1>
      <p>
        Shubble is the latest shuttle tracker, which is built using Mapkit JS, React, and Flask.
      </p>
      <p>
        Shubble is an open source project under the Rensselaer Center for Open Source (RCOS).
      </p>
      <p>
        Have an idea to improve it? Contributions are welcome. Visit our <a href='https://github.com/wtg/shubble' target='_blank'>Github Repository</a> to learn more.
      </p>
      <p>
        Interested in Shubble's data? Take a look at our
        <Link to='/data'>
          <span className = 'link1'>data page</span>
        </Link>.
      </p>
      <div className='small'>
        <p>
          &copy; 2025 SHUBBLE
        </p>
      </div>
    </div>
  )
}
