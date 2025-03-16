import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { FixedSizeGrid as Grid } from 'react-window';
import './ImageGrid.css';

interface ImageGridProps {
    rows: number;
    cols: number;
    totalRows: number;
    getImage: (row: number, col: number) => Promise<string>;
    firstVisibleRow: number;
    onScroll: (row: number) => void;
}


export function ImageGrid(props: ImageGridProps) {
    const { rows, cols, totalRows, getImage, firstVisibleRow, onScroll } = props;

    const Cell = ({ columnIndex, rowIndex, style }: {columnIndex: number, rowIndex: number, style: any}) => {
        const { data, error, isLoading } = useQuery(
            {
                queryKey: ['image', rowIndex, columnIndex],
                queryFn: async () => await getImage(rowIndex, columnIndex)
            }
        );

        if (isLoading) return <div style={style}>Loading...</div>;
        if (error || !data) return <div style={style}>Error</div>;

        return <img style={style} src={data} alt={`Image ${rowIndex}-${columnIndex}`} />;
    };

    return (
        <Grid
            columnCount={cols}
            columnWidth={100}
            height={600}
            rowCount={totalRows}
            rowHeight={100}
            width={800}
            onScroll={({ scrollTop }) => onScroll(Math.floor(scrollTop / 100))}
        >
            {Cell}
        </Grid>
    );
}