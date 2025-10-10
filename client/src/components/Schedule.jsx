import { useState, useEffect, useLayoutEffect } from 'react';
import '../styles/Schedule.css';
import scheduleData from '../data/schedule.json';
import routeData from '../data/routes.json';
import { aggregatedSchedule } from '../data/parseSchedule';

export default function Schedule({ selectedRoute, setSelectedRoute, selectedStop, setSelectedStop, collapsibleSecondaryTimeline = true, defaultExpanded = true }) {
  // Validate props once at the top
  if (typeof setSelectedRoute !== 'function') {
    throw new Error('setSelectedRoute must be a function');
  }
  if (typeof setSelectedStop !== 'function') {
    throw new Error('setSelectedStop must be a function');
  }

  const now = new Date();
  const [selectedDay, setSelectedDay] = useState(now.getDay());
  const [routeNames, setRouteNames] = useState(Object.keys(aggregatedSchedule[selectedDay]));
  const [stopNames, setStopNames] = useState([]);
  const [schedule, setSchedule] = useState([]);
  const [expandedGroups, setExpandedGroups] = useState(new Set());
  const [userHasInteracted, setUserHasInteracted] = useState(false);

  // Define safe values to avoid repeated null checks
  const safeSelectedStop = selectedStop || "all";
  const safeSelectedRoute = selectedRoute || routeNames[0];

  // Update schedule and routeNames when selectedDay changes
  useEffect(() => {
    setSchedule(aggregatedSchedule[selectedDay]);
    setRouteNames(Object.keys(aggregatedSchedule[selectedDay]));
    // If parent hasn't provided a selectedRoute yet, pick the first available one
    const firstRoute = Object.keys(aggregatedSchedule[selectedDay])[0];
    if (!selectedRoute || !(selectedRoute in aggregatedSchedule[selectedDay])) {
      setSelectedRoute(firstRoute);
    }
  }, [selectedDay, selectedRoute, setSelectedRoute]);

  // Initialize expanded groups based on defaultExpanded setting and auto-collapse past routes
  // Only run this when route or day changes, not on every render
  useEffect(() => {
    if (collapsibleSecondaryTimeline && schedule[safeSelectedRoute] && !userHasInteracted) {
      const newExpandedGroups = new Set();
      
      schedule[safeSelectedRoute].forEach((time, index) => {
        const firstStop = routeData[safeSelectedRoute].STOPS[0];
        const firstStopTime = offsetTime(time, routeData[safeSelectedRoute][firstStop].OFFSET);
        const isPastRoute = selectedDay === now.getDay() && firstStopTime < now;
        const hasUpcomingSecondary = hasUpcomingSecondaryStops(index);
        
        // Only expand if:
        // 1. defaultExpanded is true AND (it's not a past route OR it has upcoming secondary stops), OR
        // 2. It has upcoming secondary stops (keep open to show them)
        if (defaultExpanded && (!isPastRoute || hasUpcomingSecondary)) {
          newExpandedGroups.add(index);
        } else if (hasUpcomingSecondary) {
          // Always keep open if there are upcoming secondary stops
          newExpandedGroups.add(index);
        }
      });
      
      setExpandedGroups(newExpandedGroups);
    } else if (collapsibleSecondaryTimeline && !defaultExpanded && !userHasInteracted) {
      // Initialize all groups as collapsed if defaultExpanded is false
      setExpandedGroups(new Set());
    }
  }, [schedule, safeSelectedRoute, collapsibleSecondaryTimeline, defaultExpanded, selectedDay, userHasInteracted]);

  // Reset user interaction flag when route or day changes
  useEffect(() => {
    setUserHasInteracted(false);
  }, [selectedRoute, selectedDay]);

  // Update stopNames and selectedStop when selectedRoute changes
  useEffect(() => {
    if (!safeSelectedRoute || !(safeSelectedRoute in routeData)) return;
    if (!(selectedStop in routeData[safeSelectedRoute])) {
      setSelectedStop("all");
    }
    setStopNames(routeData[safeSelectedRoute].STOPS);
  }, [selectedRoute]);

  // Handle day change from dropdown
  const handleDayChange = (e) => {
    setSelectedDay(parseInt(e.target.value));
  }

  // Function to offset schedule time by given minutes
  const offsetTime = (time, offset) => {
    const date = new Date(time);
    date.setMinutes(date.getMinutes() + offset);
    return date;
  }

  // Function to toggle group expansion
  const toggleGroup = (groupIndex) => {
    if (!collapsibleSecondaryTimeline) return;
    
    setUserHasInteracted(true); // Mark that user has manually interacted
    
    const newExpandedGroups = new Set(expandedGroups);
    if (newExpandedGroups.has(groupIndex)) {
      newExpandedGroups.delete(groupIndex);
    } else {
      newExpandedGroups.add(groupIndex);
    }
    setExpandedGroups(newExpandedGroups);
  }

  // Function to toggle all groups
  const toggleAllGroups = () => {
    if (!collapsibleSecondaryTimeline || !schedule[safeSelectedRoute]) return;
    
    setUserHasInteracted(true); // Mark that user has manually interacted
    
    const allGroups = new Set();
    const hasExpandedGroups = expandedGroups.size > 0;
    
    if (!hasExpandedGroups) {
      // Expand all groups (except past routes if we want to keep them collapsed)
      schedule[safeSelectedRoute].forEach((time, index) => {
        const firstStop = routeData[safeSelectedRoute].STOPS[0];
        const firstStopTime = offsetTime(time, routeData[safeSelectedRoute][firstStop].OFFSET);
        const isPastRoute = selectedDay === now.getDay() && firstStopTime < now;
        
        // Only expand non-past routes
        if (!isPastRoute) {
          allGroups.add(index);
        }
      });
    }
    
    setExpandedGroups(allGroups);
  }

  // Function to find the next upcoming stop for a route group
  const getNextUpcomingStop = (routeGroupIndex) => {
    if (selectedDay !== now.getDay() || !schedule[safeSelectedRoute]) return null;
    
    const routeTimes = schedule[safeSelectedRoute];
    const currentRouteTime = routeTimes[routeGroupIndex];
    if (!currentRouteTime) return null;
    
    const stops = routeData[safeSelectedRoute].STOPS;
    const stopTimes = stops.map(stop => offsetTime(currentRouteTime, routeData[safeSelectedRoute][stop].OFFSET));
    
    // Find the first stop that hasn't passed yet
    for (let i = 0; i < stopTimes.length; i++) {
      if (stopTimes[i] > now) {
        return i; // Return the index of the next upcoming stop
      }
    }
    
    return null; // All stops have passed
  }

  // Function to check if a route group has any upcoming secondary stops
  const hasUpcomingSecondaryStops = (routeGroupIndex) => {
    if (selectedDay !== now.getDay() || !schedule[safeSelectedRoute]) return false;
    
    const routeTimes = schedule[safeSelectedRoute];
    const currentRouteTime = routeTimes[routeGroupIndex];
    if (!currentRouteTime) return false;
    
    const stops = routeData[safeSelectedRoute].STOPS;
    
    // Check if any secondary stops (index > 0) haven't passed yet
    for (let i = 1; i < stops.length; i++) {
      const stopTime = offsetTime(currentRouteTime, routeData[safeSelectedRoute][stops[i]].OFFSET);
      if (stopTime > now) {
        return true; // Found an upcoming secondary stop
      }
    }
    
    return false; // No upcoming secondary stops
  }

  // Update timeline line height to match content
  useEffect(() => {
    const updateTimelineLineHeight = () => {
      const timelineContainer = document.querySelector('.timeline-container');
      const timelineLine = document.querySelector('.timeline-line');
      const timelineContent = document.querySelector('.timeline-content');
      
      if (timelineContainer && timelineLine && timelineContent) {
        // Get the full height of the content
        const contentHeight = timelineContent.scrollHeight;
        // Set the timeline line height to match the content height
        timelineLine.style.height = `${contentHeight}px`;
      }
    };

    // Update height immediately
    updateTimelineLineHeight();

    // Update height when content changes (e.g., when groups expand/collapse)
    const timeoutId = setTimeout(updateTimelineLineHeight, 100);

    return () => clearTimeout(timeoutId);
  }, [schedule, selectedRoute, selectedDay, selectedStop, expandedGroups]);

  // scroll to the current time on route change
  useEffect(() => {
    const timelineContainer = document.querySelector('.timeline-container');
    if (!timelineContainer) return;

    if (selectedDay !== now.getDay()) return; // only scroll if viewing today's schedule
    
    // First, try to find a highlighted item (current time)
    let targetItem = Array.from(timelineContainer.querySelectorAll('.timeline-item.current-time, .secondary-timeline-item.current-time')).find(item => {
      return item.offsetParent !== null; // Make sure it's visible
    });

    // If no highlighted item, find the first current or future time item
    if (!targetItem) {
      targetItem = Array.from(timelineContainer.querySelectorAll('.timeline-item, .secondary-timeline-item')).find(item => {
        const timeElement = item.querySelector('.timeline-time, .secondary-timeline-time');
        if (!timeElement) return false;
        
        const text = timeElement.textContent.trim();
        const [timePart, meridian] = text.split(" ");
        if (!timePart || !meridian) return false;

        const [rawHours, rawMinutes] = timePart.split(":");
        let hours = parseInt(rawHours, 10);
        const minutes = parseInt(rawMinutes, 10);

        // Convert to 24h
        if (meridian.toUpperCase() === "PM" && hours < 12) {
          hours += 12;
        }
        if (meridian.toUpperCase() === "AM" && hours === 12) {
          hours = 0;
        }

        const timeDate = new Date();
        timeDate.setHours(hours, minutes, 0, 0);

        return timeDate >= now;
      });
    }

    if (targetItem) {
      // Add a small delay to ensure the timeline line height has been updated
      setTimeout(() => {
        targetItem.scrollIntoView({ behavior: "smooth", block: "center" });
      }, 150);
    }
  }, [selectedRoute, selectedDay, selectedStop, schedule, expandedGroups]);


  const daysOfTheWeek = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

  return (
    <div className="schedule-container">
      <div className="schedule-header">
        <h2>Schedule</h2>
        <div className="schedule-controls">
          <div className="control-group">
            <label htmlFor='weekday-dropdown'>Day:</label>
            <select id='weekday-dropdown' className="schedule-dropdown" value={selectedDay} onChange={handleDayChange}>
              {
                daysOfTheWeek.map((day, index) =>
                  <option key={index} value={index}>
                    {day}
                  </option>
                )
              }
            </select>
          </div>
           <div className="control-group">
             <label>Route:</label>
             <div className="route-toggle">
               {routeNames.map((route, index) => (
                 <button
                   key={index}
                   className={`route-toggle-button ${safeSelectedRoute === route ? 'active' : ''}`}
                   onClick={() => setSelectedRoute(route)}
                 >
                   {route}
                 </button>
               ))}
             </div>
           </div>
          <div className="control-group">
            <label htmlFor='stop-dropdown'>Stop:</label>
            <select id='stop-dropdown' className="schedule-dropdown" value={safeSelectedStop} onChange={(e) => setSelectedStop(e.target.value)}>
              <option value="all">All Stops</option>
              {
                stopNames.map((stop, index) =>
                  <option key={index} value={stop}>
                    {routeData[safeSelectedRoute][stop]?.NAME}
                  </option>
                )
              }
            </select>
          </div>
          {collapsibleSecondaryTimeline && safeSelectedStop === "all" && (
            <div className="control-group">
              <button 
                className="collapse-all-button"
                onClick={toggleAllGroups}
                title={expandedGroups.size > 0 ? "Collapse all" : "Expand all"}
              >
                {expandedGroups.size > 0 ? "Collapse All" : "Expand All"}
              </button>
            </div>
          )}
        </div>
      </div>
      
      <div className="timeline-container">
        <div className="timeline-line"></div>
        <div className="timeline-content">
          {
            safeSelectedStop === "all" ?
              schedule[safeSelectedRoute]?.map((time, index) => {
                const firstStop = routeData[safeSelectedRoute].STOPS[0];
                const firstStopTime = offsetTime(time, routeData[safeSelectedRoute][firstStop].OFFSET);
                const nextUpcomingStopIndex = getNextUpcomingStop(index);
                const isFirstStopCurrentTime = selectedDay === now.getDay() && 
                  firstStopTime.getHours() === now.getHours() && 
                  Math.abs(firstStopTime.getMinutes() - now.getMinutes()) <= 5 &&
                  nextUpcomingStopIndex === 0; // Only highlight if first stop is actually the next upcoming
                const isFirstStopPastTime = selectedDay === now.getDay() && firstStopTime < now;
                
                const isExpanded = expandedGroups.has(index);
                const hasSecondaryStops = routeData[safeSelectedRoute].STOPS.length > 1;
                
                return (
                  <div key={index} className="timeline-route-group">
                    {/* Main timeline - first stop only */}
                    <div className={`timeline-item first-stop ${isFirstStopCurrentTime ? 'current-time' : ''} ${isFirstStopPastTime ? 'past-time' : ''} ${collapsibleSecondaryTimeline && hasSecondaryStops ? 'collapsible-header' : ''}`} 
                         onClick={collapsibleSecondaryTimeline && hasSecondaryStops ? () => toggleGroup(index) : undefined}>
                      <div className="timeline-dot"></div>
                      <div className="timeline-content-item">
                        <div className="timeline-time">
                          {firstStopTime.toLocaleTimeString(undefined, { timeStyle: 'short' })}
                        </div>
                        <div className="timeline-stop">
                          {routeData[safeSelectedRoute][firstStop].NAME}
                        </div>
                        {collapsibleSecondaryTimeline && hasSecondaryStops && (
                          <div className="collapse-indicator">
                            <span className={`collapse-arrow ${isExpanded ? 'expanded' : ''}`}>▼</span>
                          </div>
                        )}
                      </div>
                    </div>
                    
                    {/* Secondary timeline - subsequent stops */}
                    {hasSecondaryStops && (
                      <div className={`secondary-timeline ${collapsibleSecondaryTimeline ? 'collapsible-content' : ''} ${collapsibleSecondaryTimeline && isExpanded ? 'expanded' : ''}`}>
                        {routeData[safeSelectedRoute].STOPS.slice(1).map((stop, sidx) => {
                          const stopTime = offsetTime(time, routeData[safeSelectedRoute][stop].OFFSET);
                          const actualStopIndex = sidx + 1; // +1 because we sliced off the first stop
                          const isUpcomingStop = nextUpcomingStopIndex === actualStopIndex;
                          const isPastTime = selectedDay === now.getDay() && stopTime < now;
                          
                          return (
                            <div key={`${index}-${sidx + 1}`} className={`secondary-timeline-item ${isUpcomingStop ? 'current-time' : ''} ${isPastTime ? 'past-time' : ''}`}>
                              <div className="secondary-timeline-dot"></div>
                              <div className="secondary-timeline-content">
                                <div className="secondary-timeline-time">
                                  {stopTime.toLocaleTimeString(undefined, { timeStyle: 'short' })}
                                </div>
                                <div className="secondary-timeline-stop">
                                  {routeData[safeSelectedRoute][stop].NAME}
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                );
              }) :
              schedule[safeSelectedRoute]?.map((time, index) => {
                const stopTime = offsetTime(time, routeData[safeSelectedRoute][selectedStop]?.OFFSET);
                const isCurrentTime = selectedDay === now.getDay() && 
                  stopTime.getHours() === now.getHours() && 
                  Math.abs(stopTime.getMinutes() - now.getMinutes()) <= 5;
                const isPastTime = selectedDay === now.getDay() && stopTime < now;
                
                return (
                  <div key={index} className={`timeline-item single-stop ${isCurrentTime ? 'current-time' : ''} ${isPastTime ? 'past-time' : ''}`}>
                    <div className="timeline-dot"></div>
                    <div className="timeline-content-item">
                      <div className="timeline-time">
                        {stopTime.toLocaleTimeString(undefined, { timeStyle: 'short' })}
                      </div>
                      <div className="timeline-stop">
                        {routeData[safeSelectedRoute][selectedStop]?.NAME}
                      </div>
                    </div>
                  </div>
                );
              })
          }
        </div>
      </div>
    </div>
  );
}
