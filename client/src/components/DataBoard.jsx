import "../styles/DataBoard.css"

export default function DataBoard({title, dataToDisplay}) {
    return (
	<>
	    <div className="data-board-container">
		<table className="data-board-table">
		    <thead>
			<tr>
			    <th className="data-board-table-header">
				{title}
			    </th>
			</tr>
		    </thead>
		    {!Array.isArray(dataToDisplay) || dataToDisplay.length == 0 || !Array.isArray(dataToDisplay[0]) ? (
			<tbody>
			    <tr>
				<td>
				    No data given
				</td>
			    </tr>
			</tbody>
		    ) : (
			<tbody>
			    {dataToDisplay[0].map((row, rowIndex) => (
				<tr key={rowIndex}>
				    {dataToDisplay.map((col, colIndex) => (
					<td key={colIndex}>{col[rowIndex]}</td>
				    ))}
				</tr>
			    ))}
			</tbody>
		    )}
		</table>
	    </div>
	</>
    );
}
