import "../styles/ShuttleRow.css"

export default function ShuttleRow({shuttleId, isActive, isAm}) {
    return (
	<>
	    <td>{shuttleId}</td>
	    <td>{isActive ? "Active" : "Inactive"}</td>
	    <td>{isAm ? "AM" : "PM"}</td>
	</>
    );
}
