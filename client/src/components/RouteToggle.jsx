import '../styles/RouteToggle.css';
import rawAggregatedSchedule from '../data/aggregated_schedule.json';

export default function RouteToggle({selectedRoute, setSelectedRoute}) {
   const today = new Date();
   const keys = Object.keys(rawAggregatedSchedule[today.getDay()])

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