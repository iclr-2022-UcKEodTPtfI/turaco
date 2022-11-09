fun (sunPosition[1], emission[1]) {
  ambient := sunPosition
  emission := mul(emission, 0.1)
  res := add(ambient, emission)
} return res;
