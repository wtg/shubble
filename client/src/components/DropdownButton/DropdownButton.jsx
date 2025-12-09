import "./DropdownButton.css"
// import {FaChevronDown} from "react-icons/fa"

const DropdownButton = ({children, open, toggle}) => {
    return (
        <div onClick={toggle}
        className={`dropdownbtn ${open ? 
            "button-open":null}`}
            >
                
            {children}
            <span className="toggle-icon">
                {/* <FaChevronDown /> */}
            </span>
        </div>
    )
    
};

export default DropdownButton; 