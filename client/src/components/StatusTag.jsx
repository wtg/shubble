import '../styles/StatusTag.css';

export default function StatusTag({isActive}) {
    return (
	<>
	    {isActive ? (
		<div className="active-tag">Active</div>
	    ) : (
		<div className="inactive-tag">Inactive</div>
	    )}
	</>
    );
}
