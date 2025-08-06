import "../styles/DataBoard.css"

export default function DataBoard({title, dataToDisplay}) {
    return (
	<>
	    <div className="data-board-container">
		<table className="data-board-table">
		    <thead>
			<tr className="data-board-table-row">
			    <th className="data-board-table-header">
				{title}
			    </th>
			</tr>
		    </thead>
		    <tbody>
			<tr className="data-board-table-row">
			    <td>
				{dataToDisplay.map(data => (
				    <tr>
					<td>
					    {data}
					</td>
				    </tr>
				))}
			    </td>
			</tr>
		    </tbody>
		</table>
	    </div>
	</>
    );
}
