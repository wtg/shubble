import { Link, Outlet } from "react-router";
import Feedback from "./Feedback";
import AnnouncementBanner, { Banner } from "./AnnouncementBanner";
import config from "../utils/config";

export default function Navigation({ GIT_REV }: { GIT_REV: string }) {
  // Build staging warning message with markdown link
  const stagingMessage = `This is a staging domain. [Visit our official website](https://shuttles.rpi.edu)${GIT_REV !== 'unknown' ? ` | [${GIT_REV}](https://github.com/wtg/shubble/commit/${GIT_REV})` : ''
    }`;

  return (
    <>
      <header className="title">
        <Link to='/' className='site-title'>SHUBBLE</Link>
        <nav className='big'>
          <ul>
            <li>
              <Link to='/about'>
                <span className="big">About</span>
              </Link>
            </li>
            <li>
              <Link to='/'>
                <span className='big'>Live Location</span>
              </Link>
            </li>
            <li className='schedule'>
              <Link to='/schedule'>
                <span className='big'>Schedule</span>
              </Link>
            </li>
          </ul>
        </nav>
      </header>

      <div className="App">
        {config.isStaging && <Banner message={stagingMessage} type="info" />}
        <AnnouncementBanner />
        <Outlet />

        <div className="small feedback-container">
          <Feedback />
        </div>
      </div>

      <footer>
        <nav className='small'>
          <ul>
            <li>
              <Link to='/about'>
                <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 512 512"><path fill="#444" fillRule="evenodd" d="M256 42.667C138.18 42.667 42.667 138.179 42.667 256c0 117.82 95.513 213.334 213.333 213.334c117.822 0 213.334-95.513 213.334-213.334S373.822 42.667 256 42.667m26.714 128c0 15.468-11.262 26.667-26.497 26.667c-15.851 0-26.837-11.2-26.837-26.963c0-15.15 11.283-26.37 26.837-26.37c15.235 0 26.497 11.22 26.497 26.666m-48 64h42.666v128h-42.666z" /></svg>
              </Link>
            </li>
            <li>
              <Link to='/'>
                <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 20 20"><path fill="#444" d="M8.53 3.012q.233-.016.47-.016a7 7 0 0 1 6.984 7.474a.5.5 0 1 0 .998.066q.018-.267.018-.54a8 8 0 0 0-8.536-7.982a.5.5 0 1 0 .066.998M8 4.984a.5.5 0 0 1 .522-.479a6.25 6.25 0 0 1 5.972 5.973a.5.5 0 1 1-.999.043A5.25 5.25 0 0 0 8.48 5.505A.5.5 0 0 1 8 4.984M6.039 16.397l1.11 1.102a.5.5 0 0 0 .704 0l1.135-1.127l1.696-1.689a4.502 4.502 0 1 0-6.367 0c.408.408 1.116 1.113 1.722 1.714M7.5 12.621a1.125 1.125 0 1 1 0-2.25a1.125 1.125 0 0 1 0 2.25" /></svg>
              </Link>
            </li>
            <li>
              <Link to='/schedule'>
                <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24"><path fill="#444" d="M15 13h1.5v2.82l2.44 1.41l-.75 1.3L15 16.69zm4-5H5v11h4.67c-.43-.91-.67-1.93-.67-3a7 7 0 0 1 7-7c1.07 0 2.09.24 3 .67zM5 21a2 2 0 0 1-2-2V5c0-1.11.89-2 2-2h1V1h2v2h8V1h2v2h1a2 2 0 0 1 2 2v6.1c1.24 1.26 2 2.99 2 4.9a7 7 0 0 1-7 7c-1.91 0-3.64-.76-4.9-2zm11-9.85A4.85 4.85 0 0 0 11.15 16c0 2.68 2.17 4.85 4.85 4.85A4.85 4.85 0 0 0 20.85 16c0-2.68-2.17-4.85-4.85-4.85" /></svg>              </Link>
            </li>
          </ul>
        </nav>
        <div className='big'>
          <div className='big-footer'>
            <div className='git-copy'>
              <a href='https://github.com/wtg/shubble' target='_blank'>
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><path fill="#000000" d="M12 2A10 10 0 0 0 2 12c0 4.42 2.87 8.17 6.84 9.5c.5.08.66-.23.66-.5v-1.69c-2.77.6-3.36-1.34-3.36-1.34c-.46-1.16-1.11-1.47-1.11-1.47c-.91-.62.07-.6.07-.6c1 .07 1.53 1.03 1.53 1.03c.87 1.52 2.34 1.07 2.91.83c.09-.65.35-1.09.63-1.34c-2.22-.25-4.55-1.11-4.55-4.92c0-1.11.38-2 1.03-2.71c-.1-.25-.45-1.29.1-2.64c0 0 .84-.27 2.75 1.02c.79-.22 1.65-.33 2.5-.33s1.71.11 2.5.33c1.91-1.29 2.75-1.02 2.75-1.02c.55 1.35.2 2.39.1 2.64c.65.71 1.03 1.6 1.03 2.71c0 3.82-2.34 4.66-4.57 4.91c.36.31.69.92.69 1.85V21c0 .27.16.59.67.5C19.14 20.16 22 16.42 22 12A10 10 0 0 0 12 2" /></svg>
              </a>
              <p>&copy; 2025 SHUBBLE, an RCOS Project</p>
            </div>
            <Feedback />
          </div>
        </div>
      </footer>
    </>
  )
}
