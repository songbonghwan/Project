// 통합된 JSON 데이터 로드 함수
async function fetchRegionData() {
  try {
    const response = await fetch('regions.json');
    if (!response.ok) {
      throw new Error('데이터를 가져오는 데 실패했습니다.');
    }
    return await response.json();
  } catch (error) {
    console.error('에러:', error.message);
    return {};
  }
}

// 기존 마커 삭제 함수
function clearMarkers() {
  const markers = document.querySelectorAll('.marker');
  markers.forEach(marker => marker.remove());
}

// 마커 클릭 이벤트 처리
function handleMarkerClick(region) {
  // 설명 박스를 표시하고, 클릭한 지역의 정보를 업데이트
  const descriptionBox = document.getElementById('description-box');
  const regionName = document.getElementById('region-name');
  const regionValue = document.getElementById('region-value');
  const regionInfo = document.getElementById('region-info');
  const progressBar = document.getElementById('progress-bar');
  const progressValue = document.getElementById('progress-value');

  regionName.textContent = region.name;
  regionValue.textContent = `소멸 위험도: ${region.value}%`;
  regionInfo.textContent = `인구 예측: ${region.info || '정보 없음'}`;

  // 최소 10% 이상으로 표시 (0%부터 10% 사이일 경우)
  const displayValue = region.value < 10 ? 10 : region.value;

  // 막대기 색상 변경 (value에 따라 색상 설정)
  let barColor = '';
  if (region.value >= 70) {
      barColor = '#ff0000'; // 70 ~ 100은 빨간색
  } else if (region.value >= 50) {
      barColor = '#ff4848'; // 50 ~ 70은 연한 빨간색
  } else if (region.value >= 40) {
      barColor = '#ff6666'; // 40 ~ 50은 더 연한 빨간색
  } else if (region.value >= 30) {
      barColor = '#ff8282'; // 30 ~ 40은 붉은 핑크색
  } else if (region.value >= 20) {
      barColor = '#ffa3a3'; // 20 ~ 30은 연한 핑크색
  } else if (region.value >= 10) {
      barColor = '#ffbdbd'; // 10 ~ 20은 더 연한 핑크색
  } else {
      barColor = '#ffd4d4'; // 0 ~ 10은 매우 연한 핑크색
  }

  // 수치에 맞는 막대기 길이 설정
  progressBar.style.width = `${displayValue}%`;

  // 막대기 색상 설정
  progressBar.style.backgroundColor = barColor;

  // 막대기에 수치 텍스트 추가
  progressValue.textContent = `${region.value}%`;

  // 설명 박스를 표시
  descriptionBox.style.display = 'block';
}






// 마커 추가 함수
function addMarkers(regions) {
  const mapContainer = document.getElementById('image-container');
  clearMarkers(); // 기존 마커 삭제

  regions.forEach(region => {
    const marker = document.createElement('div');
    marker.className = 'marker';
    marker.style.top = region.top;
    marker.style.left = region.left;
    marker.innerHTML = `${region.name}<br>${region.value}%`;
    marker.setAttribute('data-info', region.info || '추가 정보 없음'); // 추가 정보 설정
    mapContainer.appendChild(marker);

    // 마커 클릭 이벤트 리스너 추가
    marker.addEventListener('click', () => handleMarkerClick(region));
  });
}



// 초기화 함수
async function initMap() {
  const yearSelect = document.getElementById('year-select');
  const data = await fetchRegionData(); // 전체 데이터 로드
  const regions = data[yearSelect.value]; // 기본 연도 데이터 로드
  addMarkers(regions);

  // 연도 변경 이벤트 처리
  yearSelect.addEventListener('change', () => {
    const selectedRegions = data[yearSelect.value];
    addMarkers(selectedRegions);
  });
}

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', initMap);