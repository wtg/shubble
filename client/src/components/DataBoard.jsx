import "../styles/DataBoard.css"

export default function DataBoard({title, children, numColumns}) {
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
		    <tbody className={numColumns == 1 ? "single-column-tbody" : "multi-column-tbody"}>
			<tr className="data-board-table-row">
			    <td>
				{children}
			    </td>
			</tr>
		    </tbody>
		</table>
	    </div>
	</>
    );
}
