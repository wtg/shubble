import StatusTag from '../components/StatusTag';

export default function ShuttleRow({shuttleId, isActive, isAm}) {
    return (
	<>
	    <td>{shuttleId}</td>
	    <td>
		<StatusTag
		isActive={isActive}
		/>
	    </td>
	    <td>{isAm ? "AM" : "PM"}</td>
	</>
    );
}
