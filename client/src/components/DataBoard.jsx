import "../styles/DataBoard.css"

export default function DataBoard({title, datatable}) {
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
					{!Array.isArray(datatable) || datatable.length == 0 || !Array.isArray(datatable[0])
					? (
						<tbody>
							<tr>
								<td>
									No data given
								</td>
							</tr>
						</tbody>
					) : (
						<tbody>
							{datatable.map((row, rowIndex) => (
								<tr key={rowIndex}>
									{row.map((col, colIndex) => (
										<td key={colIndex}>{col}</td>
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
