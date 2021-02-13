// 쿼리 결과 값 저장 변수
dataset = [];
dataset1 = [];

console.log(xValues);

for(var i=0; i < xValues[0].length; i++) {
    dataset.push({'name': xValues[0][i], 'value': yValues[0][i]});
}
for(var i=0; i < xValues[1].length; i++) {
    dataset1.push({'name': xValues[1][i], 'value': yValues[1][i]});
}


// x축 ticks 설정
xvalue = [];
if(xValues[0].length >= xValues[1].length) {
    for(var i = 0; i < xValues[0].length; i++) {
        xvalue.push({'name' : xValues[0][i]});
    }
} 
else {
    for(var i = 0; i < xValues[1].length; i++) {
        xvalue.push({'name' : xValues[1][i]});
    }
}

// y축 ticks 설정
data_max = [];
data_max.push({'value' : d3.max(dataset, d => d.value)});
data_max.push({'value' : d3.max(dataset1, d => d.value)});

data_min = [];
data_min.push({'value' : d3.min(dataset, d => d.value)});
data_min.push({'value' : d3.min(dataset1, d => d.value)});


// Width &  Height
var margin = { top: 30, right: 30, bottom: 50, left: 50 };  // 여백
var width = 400; // 넓이
var height = 400;  //높이

// Scale
var xScale = d3.scaleBand()
    .domain(xvalue.map(d => d.name))
    // .domain(x.map(d => d.name))
    .range([margin.left, width - margin.right])
    .padding(0.4);

var yScale = d3.scaleLinear()
    .domain([d3.min(data_min, d => (d.value/100)*100), d3.max(data_max, d => (d.value/100)*100)]).nice()
    .range([height - margin.bottom, margin.top]);


// SVG
var svg = d3.select('.visual-content')
    .append('svg')
    .attr('width', width)
    .attr('height', height)
    .attr("preserveAspectRatio", "xMinYMin meet") // 반응형
    .attr("viewBox", "0 0 " +width+' '+height); // 위치

// Draw line
var line = d3
    .line()
    .x(function (d) {
        return xScale(d.name);
    })  // x축 데이터
    .y(function (d) {
        return yScale(d.value);
    }); // y축 데이터
svg
    .append("path")
    .datum(dataset)
    .attr("fill", "none")
    .attr("stroke", "steelblue")
    .attr("stroke-width", 1.5)  // 선굵기
    .attr("d", line); // 위에 선언한 라인 그리기

// Draw line1
var line1 = d3
    .line()
    .x(function (d) {
        return xScale(d.name);
    })  // x축 데이터
    .y(function (d) {
        return yScale(d.value);
    }); // y축 데이터

svg
    .append("path")
    .datum(dataset1)
    .attr("fill", "none")
    .attr("stroke", "red")
    .attr("stroke-width", 1.5)  // 선굵기
    .attr("d", line1); // 위에 선언한 라인 그리기

// xAxis
svg // 축의 선
    .append("g")
    .attr("transform", `translate(0, ${height - margin.bottom})`) //위치
    .call(
        d3.axisBottom(xScale) // axisBottom(구분기준)
        .tickSizeOuter(0)
    );


// yAxis
svg
    .append("g")
    .attr('transform', `translate(${margin.left}, 0)`)
    .call(d3.axisLeft(yScale));

// 범례 표시
colors = ["steelblue", "red"];
var legend = svg.append("g")
    .attr("text-anchor", "end")
    .selectAll("g")
    .data(legend_value)
    .enter().append("g")
    .attr("transform", function(d, i) { return "translate(0," + i * 20 + ")"; });

legend.append("rect")
    .attr("x", width - margin.right - 5)
    .attr("y", 10)
    .attr("width", 20)
    .attr("height", 13)
    .attr("fill", function(d, i) { return colors[i]; });

legend.append("text")
    .attr("x", width - margin.right - 15)
    .attr("y", 20)
    .attr("dy", "0.01em")
    .attr("font-size", "12px")
    .text(function(d) { return d; });