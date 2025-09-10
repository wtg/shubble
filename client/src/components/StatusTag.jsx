import '../styles/StatusTag.css';

export default function StatusTag({ isActive }) {
  return (
    <>
      {isActive ? (
        <div className="active-tag">active</div>
      ) : (
        <div className="inactive-tag">inactive</div>
      )}
    </>
  );
}
