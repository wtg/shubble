import StatusTag from '../components/StatusTag';
import TimeTag from '../components/TimeTag';

export default function ShuttleRow({ shuttleId, isActive, isAm }) {
    return (
	<>
	    <td>{shuttleId}</td>
	    <td>
			<StatusTag isActive={isActive} />
	    </td>
	    <td>
			<TimeTag isAm={isAm} />
	    </td>
	</>
    );
}
