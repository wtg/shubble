import "../styles/DataBoard.css"

export default function DataBoard({title, dataToDisplay}) {
    return (
	<>
	    {!Array.isArray(dataToDisplay) || dataToDisplay.length == 0 ? (
		<div>No data given</div>
	    ) : (
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
			    {dataToDisplay[0].map((row, rowIndex) => (
				<tr key={rowIndex} className="data-board-table-row">
				    {dataToDisplay.map((col, colIndex) => (
					<td key={colIndex}>{col[rowIndex]}</td>
				    ))}
				</tr>
			    ))}
			</tbody>
		    </table>
		</div>
	    )}
	</>
    );
}
