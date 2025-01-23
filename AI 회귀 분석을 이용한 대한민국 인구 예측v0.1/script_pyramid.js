// script.js

// 차트 데이터 (각각 2025년, 2035년, 2045년, 2055년, 2065년, 2075년)
const chartData = {
    labels: ['0-9세', '10-19세', '20-29세', '30-39세', '40-49세', '50-59세', '60-69세', '70-79세', '80-89세', '90세 이상'],
    datasets: [
        {
            label: '2025',
            data: [2889658, 4532076, 5814365, 6967543, 7712135, 8697751, 8095942, 4761866, 2721648, 798105],
            backgroundColor: '#3788ca',
        },
        {
            label: '2035',
            data: [1743962, 2863651, 4523011, 5796921, 6932705, 7673574, 8523795, 7747816, 4195203, 1804294],
            backgroundColor: '#4f9e6b',
        },
        {
            label: '2045',
            data: [1166584, 1728266, 2857923, 4509441, 5767936, 6898041, 7520102, 8157271, 6825825, 2896565],
            backgroundColor: '#f39c12',
        },
        {
            label: '2055',
            data: [775583, 1156084, 1724809, 2849349, 4486893, 5739096, 6760080, 7196737, 7186555, 4705007],
            backgroundColor: '#e74c3c',
        },
        {
            label: '2065',
            data: [515633, 768602, 1153771, 1719634, 2835102, 4464458, 5624314, 6469396, 6340325, 5286385],
            backgroundColor: '#9b59b6',
        },
        {
            label: '2075',
            data: [342809, 510992, 767064, 1150309, 1711035, 2820926, 4375168, 5382468, 5699537, 4892119],
            backgroundColor: '#3498db',
        },
    ]
};

// 차트 생성 함수
function createChart(canvasId, dataIndex) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    if (window.myChart) {
        window.myChart.destroy();  // 기존 차트가 있으면 삭제
    }

    const newChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: chartData.labels,
            datasets: [{
                label: chartData.datasets[dataIndex].label,
                data: chartData.datasets[dataIndex].data,
                backgroundColor: chartData.datasets[dataIndex].backgroundColor,
            }]
        },
        options: {
            indexAxis: 'y',  // 가로 막대 그래프 설정
            responsive: true,
            scales: {
                x: {
                    beginAtZero: true,  // x축이 0부터 시작
                },
                y: {
                    reverse: true,  // y축을 역순으로 설정하여 0-9세가 아래쪽에 위치하도록
                },
            },
        }
    });

    window.myChart = newChart;  // 생성된 차트를 window.myChart에 저장
}

// 차트 선택 시 변경 처리
document.getElementById('year-select').addEventListener('change', (event) => {
    const selectedChart = event.target.value;

    // 모든 차트를 숨기기
    const allCharts = document.querySelectorAll('canvas');
    allCharts.forEach(chart => chart.classList.remove('chart-visible'));  // 기존에 보이는 차트 숨김

    // 선택된 차트만 보이기
    const selectedCanvas = document.getElementById(selectedChart);
    selectedCanvas.classList.add('chart-visible');  // 선택된 차트를 보이게

    const chartIndex = parseInt(selectedChart.replace('chart', '')) - 1; // 선택된 차트의 인덱스 계산
    createChart(selectedChart, chartIndex);  // 선택된 차트 생성
});

// 초기 차트 설정 (기본적으로 첫 번째 차트)
document.addEventListener('DOMContentLoaded', () => {
    const firstChart = document.getElementById('chart1');
    firstChart.classList.add('chart-visible');  // 기본 차트 보이기
    createChart('chart1', 0);  // 2025년 차트 기본 생성
});
