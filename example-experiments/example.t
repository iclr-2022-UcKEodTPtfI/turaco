fun (
  sunPosition, emission
) {

  if (sunPosition < 0) {
    ambient := 0
  } else {
    ambient = sunPosition;
  }

  if (sunPosition < 0.1) {
    emission := mul(emission, 0.1)
  } else {
    emission := mul(emission , sunPosition);
  }
  res := add(ambient, emission)
}  return res;
