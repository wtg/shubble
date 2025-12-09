import React, { useState } from "react";
import DropdownButton from "../DropdownButton/DropdownButton";
import DropdownContent from "../DropdownContent/DropdownContent";
import "./Dropdown.css";

const Dropdown = ({ buttonText, content }) => {
  const [open, setOpen] = useState(false);
  const toggleDropdown = () => setOpen(!open);
  const closeDropdown = () => setOpen(false);

  return (
    <div className="dropdown">
      <DropdownButton toggle={toggleDropdown} open={open}>
        {buttonText}
      </DropdownButton>

      <DropdownContent open={open}>
        {/* ✅ Wrap the content with a click handler */}
        <div
          onClick={(e) => {
            // only close when clicking a link, not when toggling dropdown
            if (e.target.tagName === "A") {
              closeDropdown();
            }
          }}
        >
          {content}
        </div>
      </DropdownContent>
    </div>
  );
};

export default Dropdown;
