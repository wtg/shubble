import '../styles/StatusTag.css';

type StatusTagProps = {
  isActive: boolean;
};

export default function StatusTag({ isActive }: StatusTagProps) {
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
