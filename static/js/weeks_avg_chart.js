// 쿼리 결과 값 저장 변수
var data = [];
console.log(xValues);
for(var i in xValues) {
  data.push({'name': xValues[i], 'value': yValues[i]});
  
}

// Width & Height
var width = 400;
var height = 400;
var margin = {top: 50, left: 50, bottom: 50, right: 30};

// Scale
var xScale = d3.scaleBand()
  .domain(data.map(d => d.name))
  .range([margin.left, width - margin.right])
  .padding(0.4);

var yScale = d3.scaleLinear()
  .domain([0, d3.max(data, d => (d.value/100)*100)]).nice()
  .range([height - margin.bottom, margin.top]);
 
// xAxis
var xAxis = g => g
  .attr('transform', `translate(0, ${height - margin.bottom})`)
  .call(d3.axisBottom(xScale)
  .tickSizeOuter(0));

// yAxis
var yAxis = g => g
  .attr('transform', `translate(${margin.left}, 0)`)
  .call(d3.axisLeft(yScale));

 
// SVG
var svg = d3.select('div.visual-content')
  .append('svg')
  .attr('width', width)
  .attr('height', height)
  .attr("preserveAspectRatio", "xMinYMin meet")
  .attr("viewBox", "0 0 " +width+' '+height);

// Draw barchart
svg.append('g').call(xAxis)
  .selectAll("text")  
  .style("text-anchor", "middle")
  .attr("dx", "-.12em")
  .attr("dy", ".8em")
  .attr("transform", "rotate(-9)");
svg.append('g').call(yAxis);
svg.append('g')
  .attr('fill', 'steelblue')
  .selectAll('rect').data(data).enter().append('rect')
  .attr('x', d => xScale(d.name))
  .attr('y', d => yScale(d.value))
  .attr('height', d => yScale(0) - yScale(d.value))
  .attr('width', xScale.bandwidth());
svg.node();

// 범례
// colors = ["steelblue"];
// var legend = svg.append("g")
//     .attr("text-anchor", "end")
//     .selectAll("g")
//     .data(legend_value)
//     .enter().append("g")
//     .attr("transform", function(d, i) { return "translate(0," + i * 20 + ")"; });

// legend.append("rect")
//     .attr("x", width - margin.right - 5)
//     .attr("y", 10)
//     .attr("width", 20)
//     .attr("height", 13)
//     .attr("fill", function(d, i) { return colors[i]; });

// legend.append("text")
// .attr("x", width - margin.right - 15)
// .attr("y", 20)
// .attr("dy", "0.01em")
// .attr("font-size", "12px")
// .text(function(d) { return (d + "월"); });