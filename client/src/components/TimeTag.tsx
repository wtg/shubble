import '../styles/TimeTag.css';

type TimeTagProps = {
  isAm: boolean;
};

export default function TimeTag({ isAm }: TimeTagProps) {
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
