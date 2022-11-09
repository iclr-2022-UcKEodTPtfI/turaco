fun (sunPosition[1], emission[1]) {
  ambient := sunPosition
  emission := mul(emission, sunPosition)
  res := add(ambient, emission)
} return res;
