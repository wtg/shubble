import '../styles/TimeTag.css';

export default function TimeTag({isAm}) {
    return (
		<>
			{isAm ? (
				<div className="am-tag">AM</div>
				) : (
				<div className="pm-tag">PM</div>
			)}
		</>
    );
}
