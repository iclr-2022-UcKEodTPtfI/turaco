fun (sunPosition[1], emission[1]) {
  ambient := 0.0
  emission := mul(emission, 0.1)
  res := add(ambient, emission)
} return res;
