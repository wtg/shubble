import { useState, useEffect, useLayoutEffect } from 'react';
import '../styles/LoopToggle.css';
import scheduleData from '../data/schedule.json';
import routeData from '../data/routes.json';
import { aggregatedSchedule } from '../data/parseSchedule';

export default function LoopToggle() {

   const [active, setActive] = useState("north");
   return(
      <div class="toggle-div">
         <button 
            className={active ==="north" ? "north-on" : "north-off"}
            onClick={() => setActive("north")}
         >
            North
         </button>
         <button
            className={active === "west" ? "west-on" : "west-off"}
            onClick={() => setActive("west")}
         >
            West
         </button>
      </div>
   );
}