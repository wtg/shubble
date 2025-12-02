import StatusTag from './StatusTag';
import TimeTag from './TimeTag';

type ShuttleRowProps = {
  shuttleId: string;
  isActive: boolean;
  isAm: boolean;
};

export default function ShuttleRow({ shuttleId, isActive, isAm }: ShuttleRowProps) {
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
