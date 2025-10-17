import { useState, useEffect, useLayoutEffect } from 'react';
import '../styles/RouteToggle.css';
import scheduleData from '../data/schedule.json';
import routeData from '../data/routes.json';
import { aggregatedSchedule } from '../data/parseSchedule';

export default function RouteToggle({selectedRoute, setSelectedRoute}) {
   console.log(aggregatedSchedule);
   const today = new Date();
   const keys = Object.keys(aggregatedSchedule[today.getDay()])

   // const [active, setActive] = useState("north");
   return(
      <div class="toggle-div">
         <button 
            className={selectedRoute ===keys[0] ? "north-on" : "north-off"}
            onClick={() => setSelectedRoute(keys[0])}          
         >
            North
         </button>
         <button
            className={selectedRoute === keys[1] ? "west-on" : "west-off"}
            onClick={() => setSelectedRoute(keys[1])}
         >
            West
         </button>
      </div>
   );
}