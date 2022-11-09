/*
  (C) 2019 David Lettier
  lettier.com
*/

#version 430

#define NUMBER_OF_LIGHTS    4
#define MAX_SHININESS     127.75
#define MAX_FRESNEL_POWER   5.0


struct NN {
  float weight_1[512*35];
  float bias_1[512];
  float weight_2[512*8];
  float bias_2[8];
};

layout(std430) readonly buffer nn_data {
  NN lrrrrl_data;
  NN lrrrlr_data;
  NN lrrlrr_data;
  NN rrrlrr_data;
  NN lrrrrr_data;
  NN lrrllr_data;
  NN lrrlrl_data;
  NN rrrlrl_data;
  NN rrrllr_data;
};

uniform float osg_FrameTime;

uniform vec2 pi;
uniform vec2 gamma;

uniform mat4 trans_world_to_view;
uniform mat4 trans_view_to_world;

uniform sampler2D p3d_Texture0;
uniform sampler2D p3d_Texture1;
uniform sampler2D p3d_Texture2;
uniform sampler2D flowTexture;
uniform sampler2D ssaoBlurTexture;

uniform struct
  { vec4 ambient
  ; vec4 diffuse
  ; vec4 emission
  ; vec3 specular
  ; float shininess
  ;
  } p3d_Material;

uniform struct
  { vec4 ambient
  ;
  } p3d_LightModel;

uniform struct p3d_LightSourceParameters
  { vec4 color

  ; vec4 ambient
  ; vec4 diffuse
  ; vec4 specular

  ; vec4 position

  ; vec3  spotDirection
  ; float spotExponent
  ; float spotCutoff
  ; float spotCosCutoff

  ; float constantAttenuation
  ; float linearAttenuation
  ; float quadraticAttenuation

  ; vec3 attenuation

  ; sampler2DShadow shadowMap

  ; mat4 shadowViewMatrix
  ;
  } p3d_LightSource[NUMBER_OF_LIGHTS];

uniform vec2 normalMapsEnabled;
uniform vec2 fresnelEnabled;
uniform vec2 rimLightEnabled;
uniform vec2 blinnPhongEnabled;
uniform vec2 celShadingEnabled;
uniform vec2 flowMapsEnabled;
uniform vec2 specularOnly;
uniform vec2 isParticle;
uniform vec2 isWater;
uniform vec2 sunPosition;

in vec4 vertexColor;

in vec4 vertexInShadowSpaces[NUMBER_OF_LIGHTS];

in vec4 vertexPosition;

in vec3 vertexNormal;
in vec3 binormal;
in vec3 tangent;

in vec2 diffuseCoord;
in vec2 normalCoord;

out vec4 out0;
out vec4 out1;

float relu(float x) {
  return max(x, 0);
}

float mclamp(float x) {
  if (x > 0.99) {
    x = 1;
  } else if (x < 0.01) {
    x = 0;
  }
  return x;
}

void main() {
  vec3  shadowColor   = pow(vec3(0.149, 0.220, 0.227), vec3(gamma.x));
  int   shadowSamples = 2;

  vec4 diffuseColor;
  if (isParticle.x == 1) {
    diffuseColor   = texture(p3d_Texture0, diffuseCoord) * vertexColor;
  } else {
    diffuseColor   = texture(p3d_Texture0, diffuseCoord);
  }
  diffuseColor.rgb = pow(diffuseColor.rgb, vec3(gamma.x));

  vec3 materialSpecularColor = p3d_Material.specular;

  vec2 flow   = texture(flowTexture, normalCoord).xy;
       flow   = (flow - 0.5) * 2.0;
       flow.x = abs(flow.x) <= 0.02 ? 0.0 : flow.x;
       flow.y = abs(flow.y) <= 0.02 ? 0.0 : flow.y;

  vec4 normalTex =
    texture
      ( p3d_Texture1
      , vec2
          ( normalCoord.x + flowMapsEnabled.x * flow.x * osg_FrameTime
          , normalCoord.y + flowMapsEnabled.y * flow.y * osg_FrameTime
          )
      );

  vec3 normal;

  if (isParticle.x == 1) {
    normal = normalize((trans_world_to_view * vec4(0.0, 0.0, 1.0, 0.0)).xyz);
  } else if (normalMapsEnabled.x == 1) {
    vec3 normalRaw =
      normalize
        ( normalTex.rgb
        * 2.0
        - 1.0
        );
    normal =
      normalize
        ( mat3
            ( tangent
            , binormal
            , vertexNormal
            )
        * normalRaw
        );
  } else {
    normal =
      normalize(vertexNormal);
  }

  vec4 specularMap = texture(p3d_Texture2, diffuseCoord);

  vec4 diffuse  = vec4(0.0, 0.0, 0.0, diffuseColor.a);
  vec4 specular = vec4(0.0, 0.0, 0.0, diffuseColor.a);

  for (int i = 0; i < p3d_LightSource.length(); ++i) {
    vec3 lightDirection =
        p3d_LightSource[i].position.xyz
      - vertexPosition.xyz
      * p3d_LightSource[i].position.w;

    vec3 unitLightDirection = normalize(lightDirection);
    vec3 eyeDirection       = normalize(-vertexPosition.xyz);
    vec3 reflectedDirection = normalize(-reflect(unitLightDirection, normal));
    vec3 halfwayDirection   = normalize(unitLightDirection + eyeDirection);

    float lightDistance = length(lightDirection);

    float attenuation =
        1.0
      / ( p3d_LightSource[i].constantAttenuation
        + p3d_LightSource[i].linearAttenuation
        * lightDistance
        + p3d_LightSource[i].quadraticAttenuation
        * (lightDistance * lightDistance)
        );

    if (attenuation <= 0.0) { continue; }

    float diffuseIntensity = dot(normal, unitLightDirection);

    if (diffuseIntensity < 0.0) { continue; }

    diffuseIntensity =
        celShadingEnabled.x == 1
      ? smoothstep(0.1, 0.2, diffuseIntensity)
      : diffuseIntensity;

    vec4 lightDiffuseColor     = p3d_LightSource[i].diffuse;
         lightDiffuseColor.rgb = pow(lightDiffuseColor.rgb, vec3(gamma.x));

    vec4 diffuseTemp =
      vec4
        ( clamp
            (   diffuseColor.rgb
              * lightDiffuseColor.rgb
              * diffuseIntensity
            , 0.0
            , 1.0
            )
        , diffuseColor.a
        );

    float specularIntensity =
      ( blinnPhongEnabled.x == 1
      ? clamp(dot(normal,       halfwayDirection),   0.0, 1.0)
      : clamp(dot(eyeDirection, reflectedDirection), 0.0, 1.0)
      );

    specularIntensity =
      ( celShadingEnabled.x == 1
      ? smoothstep(0.9, 1.0, specularIntensity)
      : specularIntensity
      );

    vec4  lightSpecularColor     = p3d_LightSource[i].specular;
          lightSpecularColor.rgb = pow(lightSpecularColor.rgb, vec3(gamma.x));

    vec4 materialSpecularColor        = vec4(vec3(specularMap.r), diffuseColor.a);
    if (fresnelEnabled.x == 1) {
      float fresnelFactor             = dot((blinnPhongEnabled.x == 1 ? halfwayDirection : normal), eyeDirection);
            fresnelFactor             = max(fresnelFactor, 0.0);
            fresnelFactor             = 1.0 - fresnelFactor;
            fresnelFactor             = pow(fresnelFactor, specularMap.b * MAX_FRESNEL_POWER);
            materialSpecularColor.rgb = mix(materialSpecularColor.rgb, vec3(1.0), clamp(fresnelFactor, 0.0, 1.0));
    }

    vec4 specularTemp      = vec4(vec3(0.0), diffuseColor.a);
         specularTemp.rgb  = lightSpecularColor.rgb * pow(specularIntensity, specularMap.g * MAX_SHININESS);
         specularTemp.rgb *= materialSpecularColor.rgb;
         specularTemp.rgb *= (1 - isParticle.x);
         specularTemp.rgb  = clamp(specularTemp.rgb, 0.0, 1.0);

    float unitLightDirectionDelta =
      dot
        ( normalize(p3d_LightSource[i].spotDirection)
        , -unitLightDirection
        );

    if (unitLightDirectionDelta < p3d_LightSource[i].spotCosCutoff) { continue; }

    float spotExponent = p3d_LightSource[i].spotExponent;

    diffuseTemp.rgb *= (spotExponent <= 0.0 ? 1.0 : pow(unitLightDirectionDelta, spotExponent));

    vec2  shadowMapSize = textureSize(p3d_LightSource[i].shadowMap, 0);
    float inShadow      = 0.0;
    float count         = 0.0;

    for (  int si = -shadowSamples; si <= shadowSamples; ++si) {
      for (int sj = -shadowSamples; sj <= shadowSamples; ++sj) {
        inShadow +=
          ( 1.0
          - textureProj
              ( p3d_LightSource[i].shadowMap
              , vertexInShadowSpaces[i] + vec4(vec2(si, sj) / shadowMapSize, vec2(0.0))
              )
          );

        count += 1.0;
      }
    }

    inShadow /= count;

    vec3 shadow =
      mix
        ( vec3(1.0)
        , shadowColor
        , inShadow
        );

    diffuseTemp.rgb  *= mix(shadow, vec3(1.0), isParticle.x);
    specularTemp.rgb *= mix(shadow, vec3(1.0), isParticle.x);

    diffuseTemp.rgb  *= attenuation;
    specularTemp.rgb *= attenuation;

    diffuse.rgb  += diffuseTemp.rgb;
    specular.rgb += specularTemp.rgb;
  }



  vec4 rimLight = vec4(vec3(0.0), diffuseColor.a);
  if (rimLightEnabled.x == 1) {
       rimLight.rgb =
        vec3
          ( 1.0
          - max
              ( 0.0
              , dot(normalize(-vertexPosition.xyz), normalize(normal))
              )
          );
       rimLight.rgb =
          ( celShadingEnabled.x == 1
          ? smoothstep(0.3, 0.4, rimLight.rgb)
          : pow(rimLight.rgb, vec3(2.0)) * 1.2
          );
       rimLight.rgb *= diffuse.rgb;
  }
  vec3 worldNormal = normalize((trans_view_to_world * vec4(normal, 0.0)).xyz);

  vec2 ssaoBlurTexSize  = textureSize(ssaoBlurTexture, 0).xy;
  vec2 ssaoBlurTexCoord = gl_FragCoord.xy / ssaoBlurTexSize;
  vec3 ssao             = texture(ssaoBlurTexture, ssaoBlurTexCoord).rgb;
       ssao             = mix(shadowColor, vec3(1.0), clamp(ssao.r, 0.0, 1.0));


  float sunPositionC  = sin(sunPosition.x * pi.y);
  float sunMixFactorC = 1.0 - (sunPositionC / 2.0 + 0.5);

  float imt[512];
  float oup[8];

  if (0.5 > sunMixFactorC || sunPosition.x >= 359.99) {
    if (0.00316228 > sunPositionC) {
      if (isWater[0] > 0) {
          for (int i = 0; i < 512; i++) {
    imt[i] = lrrllr_data.bias_1[i] + lrrllr_data.weight_1[i*35+0]*rimLight.x+lrrllr_data.weight_1[i*35+1]*rimLight.y+lrrllr_data.weight_1[i*35+2]*rimLight.z+lrrllr_data.weight_1[i*35+3]*rimLight.w+lrrllr_data.weight_1[i*35+4]*celShadingEnabled.x+lrrllr_data.weight_1[i*35+5]*celShadingEnabled.y+lrrllr_data.weight_1[i*35+6]*sunPosition.x/360+lrrllr_data.weight_1[i*35+7]*sunPosition.y+lrrllr_data.weight_1[i*35+8]*gamma.x+lrrllr_data.weight_1[i*35+9]*gamma.y+lrrllr_data.weight_1[i*35+10]*worldNormal.x+lrrllr_data.weight_1[i*35+11]*worldNormal.y+lrrllr_data.weight_1[i*35+12]*worldNormal.z+lrrllr_data.weight_1[i*35+13]*ssao.x+lrrllr_data.weight_1[i*35+14]*ssao.y+lrrllr_data.weight_1[i*35+15]*ssao.z+lrrllr_data.weight_1[i*35+16]*diffuseColor.x+lrrllr_data.weight_1[i*35+17]*diffuseColor.y+lrrllr_data.weight_1[i*35+18]*diffuseColor.z+lrrllr_data.weight_1[i*35+19]*diffuseColor.w+lrrllr_data.weight_1[i*35+20]*diffuse.x+lrrllr_data.weight_1[i*35+21]*diffuse.y+lrrllr_data.weight_1[i*35+22]*diffuse.z+lrrllr_data.weight_1[i*35+23]*diffuse.w+lrrllr_data.weight_1[i*35+24]*specular.x+lrrllr_data.weight_1[i*35+25]*specular.y+lrrllr_data.weight_1[i*35+26]*specular.z+lrrllr_data.weight_1[i*35+27]*specular.w+lrrllr_data.weight_1[i*35+28]*p3d_Material.emission.xyz.x+lrrllr_data.weight_1[i*35+29]*p3d_Material.emission.xyz.y+lrrllr_data.weight_1[i*35+30]*p3d_Material.emission.xyz.z+lrrllr_data.weight_1[i*35+31]*isWater.x+lrrllr_data.weight_1[i*35+32]*isWater.y+lrrllr_data.weight_1[i*35+33]*isParticle.x+lrrllr_data.weight_1[i*35+34]*isParticle.y;
    imt[i] = relu(imt[i]);
  }
  for (int i = 0; i < 8; i++) {
    oup[i] = lrrllr_data.bias_2[i];
    for (int j = 0; j < 512; j++) {
      oup[i] += lrrllr_data.weight_2[i*512+j] * imt[j];
    }
  }

      } else {
        if (isParticle[0] > 0) {
            for (int i = 0; i < 512; i++) {
    imt[i] = lrrlrl_data.bias_1[i] + lrrlrl_data.weight_1[i*35+0]*rimLight.x+lrrlrl_data.weight_1[i*35+1]*rimLight.y+lrrlrl_data.weight_1[i*35+2]*rimLight.z+lrrlrl_data.weight_1[i*35+3]*rimLight.w+lrrlrl_data.weight_1[i*35+4]*celShadingEnabled.x+lrrlrl_data.weight_1[i*35+5]*celShadingEnabled.y+lrrlrl_data.weight_1[i*35+6]*sunPosition.x/360+lrrlrl_data.weight_1[i*35+7]*sunPosition.y+lrrlrl_data.weight_1[i*35+8]*gamma.x+lrrlrl_data.weight_1[i*35+9]*gamma.y+lrrlrl_data.weight_1[i*35+10]*worldNormal.x+lrrlrl_data.weight_1[i*35+11]*worldNormal.y+lrrlrl_data.weight_1[i*35+12]*worldNormal.z+lrrlrl_data.weight_1[i*35+13]*ssao.x+lrrlrl_data.weight_1[i*35+14]*ssao.y+lrrlrl_data.weight_1[i*35+15]*ssao.z+lrrlrl_data.weight_1[i*35+16]*diffuseColor.x+lrrlrl_data.weight_1[i*35+17]*diffuseColor.y+lrrlrl_data.weight_1[i*35+18]*diffuseColor.z+lrrlrl_data.weight_1[i*35+19]*diffuseColor.w+lrrlrl_data.weight_1[i*35+20]*diffuse.x+lrrlrl_data.weight_1[i*35+21]*diffuse.y+lrrlrl_data.weight_1[i*35+22]*diffuse.z+lrrlrl_data.weight_1[i*35+23]*diffuse.w+lrrlrl_data.weight_1[i*35+24]*specular.x+lrrlrl_data.weight_1[i*35+25]*specular.y+lrrlrl_data.weight_1[i*35+26]*specular.z+lrrlrl_data.weight_1[i*35+27]*specular.w+lrrlrl_data.weight_1[i*35+28]*p3d_Material.emission.xyz.x+lrrlrl_data.weight_1[i*35+29]*p3d_Material.emission.xyz.y+lrrlrl_data.weight_1[i*35+30]*p3d_Material.emission.xyz.z+lrrlrl_data.weight_1[i*35+31]*isWater.x+lrrlrl_data.weight_1[i*35+32]*isWater.y+lrrlrl_data.weight_1[i*35+33]*isParticle.x+lrrlrl_data.weight_1[i*35+34]*isParticle.y;
    imt[i] = relu(imt[i]);
  }
  for (int i = 0; i < 8; i++) {
    oup[i] = lrrlrl_data.bias_2[i];
    for (int j = 0; j < 512; j++) {
      oup[i] += lrrlrl_data.weight_2[i*512+j] * imt[j];
    }
  }

        } else {
            for (int i = 0; i < 512; i++) {
    imt[i] = lrrlrr_data.bias_1[i] + lrrlrr_data.weight_1[i*35+0]*rimLight.x+lrrlrr_data.weight_1[i*35+1]*rimLight.y+lrrlrr_data.weight_1[i*35+2]*rimLight.z+lrrlrr_data.weight_1[i*35+3]*rimLight.w+lrrlrr_data.weight_1[i*35+4]*celShadingEnabled.x+lrrlrr_data.weight_1[i*35+5]*celShadingEnabled.y+lrrlrr_data.weight_1[i*35+6]*sunPosition.x/360+lrrlrr_data.weight_1[i*35+7]*sunPosition.y+lrrlrr_data.weight_1[i*35+8]*gamma.x+lrrlrr_data.weight_1[i*35+9]*gamma.y+lrrlrr_data.weight_1[i*35+10]*worldNormal.x+lrrlrr_data.weight_1[i*35+11]*worldNormal.y+lrrlrr_data.weight_1[i*35+12]*worldNormal.z+lrrlrr_data.weight_1[i*35+13]*ssao.x+lrrlrr_data.weight_1[i*35+14]*ssao.y+lrrlrr_data.weight_1[i*35+15]*ssao.z+lrrlrr_data.weight_1[i*35+16]*diffuseColor.x+lrrlrr_data.weight_1[i*35+17]*diffuseColor.y+lrrlrr_data.weight_1[i*35+18]*diffuseColor.z+lrrlrr_data.weight_1[i*35+19]*diffuseColor.w+lrrlrr_data.weight_1[i*35+20]*diffuse.x+lrrlrr_data.weight_1[i*35+21]*diffuse.y+lrrlrr_data.weight_1[i*35+22]*diffuse.z+lrrlrr_data.weight_1[i*35+23]*diffuse.w+lrrlrr_data.weight_1[i*35+24]*specular.x+lrrlrr_data.weight_1[i*35+25]*specular.y+lrrlrr_data.weight_1[i*35+26]*specular.z+lrrlrr_data.weight_1[i*35+27]*specular.w+lrrlrr_data.weight_1[i*35+28]*p3d_Material.emission.xyz.x+lrrlrr_data.weight_1[i*35+29]*p3d_Material.emission.xyz.y+lrrlrr_data.weight_1[i*35+30]*p3d_Material.emission.xyz.z+lrrlrr_data.weight_1[i*35+31]*isWater.x+lrrlrr_data.weight_1[i*35+32]*isWater.y+lrrlrr_data.weight_1[i*35+33]*isParticle.x+lrrlrr_data.weight_1[i*35+34]*isParticle.y;
    imt[i] = relu(imt[i]);
  }
  for (int i = 0; i < 8; i++) {
    oup[i] = lrrlrr_data.bias_2[i];
    for (int j = 0; j < 512; j++) {
      oup[i] += lrrlrr_data.weight_2[i*512+j] * imt[j];
    }
  }

        }
      }
    } else {
      if (isWater[0] > 0) {
          for (int i = 0; i < 512; i++) {
    imt[i] = lrrrlr_data.bias_1[i] + lrrrlr_data.weight_1[i*35+0]*rimLight.x+lrrrlr_data.weight_1[i*35+1]*rimLight.y+lrrrlr_data.weight_1[i*35+2]*rimLight.z+lrrrlr_data.weight_1[i*35+3]*rimLight.w+lrrrlr_data.weight_1[i*35+4]*celShadingEnabled.x+lrrrlr_data.weight_1[i*35+5]*celShadingEnabled.y+lrrrlr_data.weight_1[i*35+6]*sunPosition.x/360+lrrrlr_data.weight_1[i*35+7]*sunPosition.y+lrrrlr_data.weight_1[i*35+8]*gamma.x+lrrrlr_data.weight_1[i*35+9]*gamma.y+lrrrlr_data.weight_1[i*35+10]*worldNormal.x+lrrrlr_data.weight_1[i*35+11]*worldNormal.y+lrrrlr_data.weight_1[i*35+12]*worldNormal.z+lrrrlr_data.weight_1[i*35+13]*ssao.x+lrrrlr_data.weight_1[i*35+14]*ssao.y+lrrrlr_data.weight_1[i*35+15]*ssao.z+lrrrlr_data.weight_1[i*35+16]*diffuseColor.x+lrrrlr_data.weight_1[i*35+17]*diffuseColor.y+lrrrlr_data.weight_1[i*35+18]*diffuseColor.z+lrrrlr_data.weight_1[i*35+19]*diffuseColor.w+lrrrlr_data.weight_1[i*35+20]*diffuse.x+lrrrlr_data.weight_1[i*35+21]*diffuse.y+lrrrlr_data.weight_1[i*35+22]*diffuse.z+lrrrlr_data.weight_1[i*35+23]*diffuse.w+lrrrlr_data.weight_1[i*35+24]*specular.x+lrrrlr_data.weight_1[i*35+25]*specular.y+lrrrlr_data.weight_1[i*35+26]*specular.z+lrrrlr_data.weight_1[i*35+27]*specular.w+lrrrlr_data.weight_1[i*35+28]*p3d_Material.emission.xyz.x+lrrrlr_data.weight_1[i*35+29]*p3d_Material.emission.xyz.y+lrrrlr_data.weight_1[i*35+30]*p3d_Material.emission.xyz.z+lrrrlr_data.weight_1[i*35+31]*isWater.x+lrrrlr_data.weight_1[i*35+32]*isWater.y+lrrrlr_data.weight_1[i*35+33]*isParticle.x+lrrrlr_data.weight_1[i*35+34]*isParticle.y;
    imt[i] = relu(imt[i]);
  }
  for (int i = 0; i < 8; i++) {
    oup[i] = lrrrlr_data.bias_2[i];
    for (int j = 0; j < 512; j++) {
      oup[i] += lrrrlr_data.weight_2[i*512+j] * imt[j];
    }
  }

      } else {
        if (isParticle[0] > 0) {
            for (int i = 0; i < 512; i++) {
    imt[i] = lrrrrl_data.bias_1[i] + lrrrrl_data.weight_1[i*35+0]*rimLight.x+lrrrrl_data.weight_1[i*35+1]*rimLight.y+lrrrrl_data.weight_1[i*35+2]*rimLight.z+lrrrrl_data.weight_1[i*35+3]*rimLight.w+lrrrrl_data.weight_1[i*35+4]*celShadingEnabled.x+lrrrrl_data.weight_1[i*35+5]*celShadingEnabled.y+lrrrrl_data.weight_1[i*35+6]*sunPosition.x/360+lrrrrl_data.weight_1[i*35+7]*sunPosition.y+lrrrrl_data.weight_1[i*35+8]*gamma.x+lrrrrl_data.weight_1[i*35+9]*gamma.y+lrrrrl_data.weight_1[i*35+10]*worldNormal.x+lrrrrl_data.weight_1[i*35+11]*worldNormal.y+lrrrrl_data.weight_1[i*35+12]*worldNormal.z+lrrrrl_data.weight_1[i*35+13]*ssao.x+lrrrrl_data.weight_1[i*35+14]*ssao.y+lrrrrl_data.weight_1[i*35+15]*ssao.z+lrrrrl_data.weight_1[i*35+16]*diffuseColor.x+lrrrrl_data.weight_1[i*35+17]*diffuseColor.y+lrrrrl_data.weight_1[i*35+18]*diffuseColor.z+lrrrrl_data.weight_1[i*35+19]*diffuseColor.w+lrrrrl_data.weight_1[i*35+20]*diffuse.x+lrrrrl_data.weight_1[i*35+21]*diffuse.y+lrrrrl_data.weight_1[i*35+22]*diffuse.z+lrrrrl_data.weight_1[i*35+23]*diffuse.w+lrrrrl_data.weight_1[i*35+24]*specular.x+lrrrrl_data.weight_1[i*35+25]*specular.y+lrrrrl_data.weight_1[i*35+26]*specular.z+lrrrrl_data.weight_1[i*35+27]*specular.w+lrrrrl_data.weight_1[i*35+28]*p3d_Material.emission.xyz.x+lrrrrl_data.weight_1[i*35+29]*p3d_Material.emission.xyz.y+lrrrrl_data.weight_1[i*35+30]*p3d_Material.emission.xyz.z+lrrrrl_data.weight_1[i*35+31]*isWater.x+lrrrrl_data.weight_1[i*35+32]*isWater.y+lrrrrl_data.weight_1[i*35+33]*isParticle.x+lrrrrl_data.weight_1[i*35+34]*isParticle.y;
    imt[i] = relu(imt[i]);
  }
  for (int i = 0; i < 8; i++) {
    oup[i] = lrrrrl_data.bias_2[i];
    for (int j = 0; j < 512; j++) {
      oup[i] += lrrrrl_data.weight_2[i*512+j] * imt[j];
    }
  }

        } else {
            for (int i = 0; i < 512; i++) {
    imt[i] = lrrrrr_data.bias_1[i] + lrrrrr_data.weight_1[i*35+0]*rimLight.x+lrrrrr_data.weight_1[i*35+1]*rimLight.y+lrrrrr_data.weight_1[i*35+2]*rimLight.z+lrrrrr_data.weight_1[i*35+3]*rimLight.w+lrrrrr_data.weight_1[i*35+4]*celShadingEnabled.x+lrrrrr_data.weight_1[i*35+5]*celShadingEnabled.y+lrrrrr_data.weight_1[i*35+6]*sunPosition.x/360+lrrrrr_data.weight_1[i*35+7]*sunPosition.y+lrrrrr_data.weight_1[i*35+8]*gamma.x+lrrrrr_data.weight_1[i*35+9]*gamma.y+lrrrrr_data.weight_1[i*35+10]*worldNormal.x+lrrrrr_data.weight_1[i*35+11]*worldNormal.y+lrrrrr_data.weight_1[i*35+12]*worldNormal.z+lrrrrr_data.weight_1[i*35+13]*ssao.x+lrrrrr_data.weight_1[i*35+14]*ssao.y+lrrrrr_data.weight_1[i*35+15]*ssao.z+lrrrrr_data.weight_1[i*35+16]*diffuseColor.x+lrrrrr_data.weight_1[i*35+17]*diffuseColor.y+lrrrrr_data.weight_1[i*35+18]*diffuseColor.z+lrrrrr_data.weight_1[i*35+19]*diffuseColor.w+lrrrrr_data.weight_1[i*35+20]*diffuse.x+lrrrrr_data.weight_1[i*35+21]*diffuse.y+lrrrrr_data.weight_1[i*35+22]*diffuse.z+lrrrrr_data.weight_1[i*35+23]*diffuse.w+lrrrrr_data.weight_1[i*35+24]*specular.x+lrrrrr_data.weight_1[i*35+25]*specular.y+lrrrrr_data.weight_1[i*35+26]*specular.z+lrrrrr_data.weight_1[i*35+27]*specular.w+lrrrrr_data.weight_1[i*35+28]*p3d_Material.emission.xyz.x+lrrrrr_data.weight_1[i*35+29]*p3d_Material.emission.xyz.y+lrrrrr_data.weight_1[i*35+30]*p3d_Material.emission.xyz.z+lrrrrr_data.weight_1[i*35+31]*isWater.x+lrrrrr_data.weight_1[i*35+32]*isWater.y+lrrrrr_data.weight_1[i*35+33]*isParticle.x+lrrrrr_data.weight_1[i*35+34]*isParticle.y;
    imt[i] = relu(imt[i]);
  }
  for (int i = 0; i < 8; i++) {
    oup[i] = lrrrrr_data.bias_2[i];
    for (int j = 0; j < 512; j++) {
      oup[i] += lrrrrr_data.weight_2[i*512+j] * imt[j];
    }
  }

        }
      }
    }
  } else {
    if (0.00316228 > sunPositionC) {
      if (isWater[0] > 0) {
          for (int i = 0; i < 512; i++) {
    imt[i] = rrrllr_data.bias_1[i] + rrrllr_data.weight_1[i*35+0]*rimLight.x+rrrllr_data.weight_1[i*35+1]*rimLight.y+rrrllr_data.weight_1[i*35+2]*rimLight.z+rrrllr_data.weight_1[i*35+3]*rimLight.w+rrrllr_data.weight_1[i*35+4]*celShadingEnabled.x+rrrllr_data.weight_1[i*35+5]*celShadingEnabled.y+rrrllr_data.weight_1[i*35+6]*sunPosition.x/360+rrrllr_data.weight_1[i*35+7]*sunPosition.y+rrrllr_data.weight_1[i*35+8]*gamma.x+rrrllr_data.weight_1[i*35+9]*gamma.y+rrrllr_data.weight_1[i*35+10]*worldNormal.x+rrrllr_data.weight_1[i*35+11]*worldNormal.y+rrrllr_data.weight_1[i*35+12]*worldNormal.z+rrrllr_data.weight_1[i*35+13]*ssao.x+rrrllr_data.weight_1[i*35+14]*ssao.y+rrrllr_data.weight_1[i*35+15]*ssao.z+rrrllr_data.weight_1[i*35+16]*diffuseColor.x+rrrllr_data.weight_1[i*35+17]*diffuseColor.y+rrrllr_data.weight_1[i*35+18]*diffuseColor.z+rrrllr_data.weight_1[i*35+19]*diffuseColor.w+rrrllr_data.weight_1[i*35+20]*diffuse.x+rrrllr_data.weight_1[i*35+21]*diffuse.y+rrrllr_data.weight_1[i*35+22]*diffuse.z+rrrllr_data.weight_1[i*35+23]*diffuse.w+rrrllr_data.weight_1[i*35+24]*specular.x+rrrllr_data.weight_1[i*35+25]*specular.y+rrrllr_data.weight_1[i*35+26]*specular.z+rrrllr_data.weight_1[i*35+27]*specular.w+rrrllr_data.weight_1[i*35+28]*p3d_Material.emission.xyz.x+rrrllr_data.weight_1[i*35+29]*p3d_Material.emission.xyz.y+rrrllr_data.weight_1[i*35+30]*p3d_Material.emission.xyz.z+rrrllr_data.weight_1[i*35+31]*isWater.x+rrrllr_data.weight_1[i*35+32]*isWater.y+rrrllr_data.weight_1[i*35+33]*isParticle.x+rrrllr_data.weight_1[i*35+34]*isParticle.y;
    imt[i] = relu(imt[i]);
  }
  for (int i = 0; i < 8; i++) {
    oup[i] = rrrllr_data.bias_2[i];
    for (int j = 0; j < 512; j++) {
      oup[i] += rrrllr_data.weight_2[i*512+j] * imt[j];
    }
  }

      } else {
        if (isParticle[0] > 0) {
            for (int i = 0; i < 512; i++) {
    imt[i] = rrrlrl_data.bias_1[i] + rrrlrl_data.weight_1[i*35+0]*rimLight.x+rrrlrl_data.weight_1[i*35+1]*rimLight.y+rrrlrl_data.weight_1[i*35+2]*rimLight.z+rrrlrl_data.weight_1[i*35+3]*rimLight.w+rrrlrl_data.weight_1[i*35+4]*celShadingEnabled.x+rrrlrl_data.weight_1[i*35+5]*celShadingEnabled.y+rrrlrl_data.weight_1[i*35+6]*sunPosition.x/360+rrrlrl_data.weight_1[i*35+7]*sunPosition.y+rrrlrl_data.weight_1[i*35+8]*gamma.x+rrrlrl_data.weight_1[i*35+9]*gamma.y+rrrlrl_data.weight_1[i*35+10]*worldNormal.x+rrrlrl_data.weight_1[i*35+11]*worldNormal.y+rrrlrl_data.weight_1[i*35+12]*worldNormal.z+rrrlrl_data.weight_1[i*35+13]*ssao.x+rrrlrl_data.weight_1[i*35+14]*ssao.y+rrrlrl_data.weight_1[i*35+15]*ssao.z+rrrlrl_data.weight_1[i*35+16]*diffuseColor.x+rrrlrl_data.weight_1[i*35+17]*diffuseColor.y+rrrlrl_data.weight_1[i*35+18]*diffuseColor.z+rrrlrl_data.weight_1[i*35+19]*diffuseColor.w+rrrlrl_data.weight_1[i*35+20]*diffuse.x+rrrlrl_data.weight_1[i*35+21]*diffuse.y+rrrlrl_data.weight_1[i*35+22]*diffuse.z+rrrlrl_data.weight_1[i*35+23]*diffuse.w+rrrlrl_data.weight_1[i*35+24]*specular.x+rrrlrl_data.weight_1[i*35+25]*specular.y+rrrlrl_data.weight_1[i*35+26]*specular.z+rrrlrl_data.weight_1[i*35+27]*specular.w+rrrlrl_data.weight_1[i*35+28]*p3d_Material.emission.xyz.x+rrrlrl_data.weight_1[i*35+29]*p3d_Material.emission.xyz.y+rrrlrl_data.weight_1[i*35+30]*p3d_Material.emission.xyz.z+rrrlrl_data.weight_1[i*35+31]*isWater.x+rrrlrl_data.weight_1[i*35+32]*isWater.y+rrrlrl_data.weight_1[i*35+33]*isParticle.x+rrrlrl_data.weight_1[i*35+34]*isParticle.y;
    imt[i] = relu(imt[i]);
  }
  for (int i = 0; i < 8; i++) {
    oup[i] = rrrlrl_data.bias_2[i];
    for (int j = 0; j < 512; j++) {
      oup[i] += rrrlrl_data.weight_2[i*512+j] * imt[j];
    }
  }

        } else {
            for (int i = 0; i < 512; i++) {
    imt[i] = rrrlrr_data.bias_1[i] + rrrlrr_data.weight_1[i*35+0]*rimLight.x+rrrlrr_data.weight_1[i*35+1]*rimLight.y+rrrlrr_data.weight_1[i*35+2]*rimLight.z+rrrlrr_data.weight_1[i*35+3]*rimLight.w+rrrlrr_data.weight_1[i*35+4]*celShadingEnabled.x+rrrlrr_data.weight_1[i*35+5]*celShadingEnabled.y+rrrlrr_data.weight_1[i*35+6]*sunPosition.x/360+rrrlrr_data.weight_1[i*35+7]*sunPosition.y+rrrlrr_data.weight_1[i*35+8]*gamma.x+rrrlrr_data.weight_1[i*35+9]*gamma.y+rrrlrr_data.weight_1[i*35+10]*worldNormal.x+rrrlrr_data.weight_1[i*35+11]*worldNormal.y+rrrlrr_data.weight_1[i*35+12]*worldNormal.z+rrrlrr_data.weight_1[i*35+13]*ssao.x+rrrlrr_data.weight_1[i*35+14]*ssao.y+rrrlrr_data.weight_1[i*35+15]*ssao.z+rrrlrr_data.weight_1[i*35+16]*diffuseColor.x+rrrlrr_data.weight_1[i*35+17]*diffuseColor.y+rrrlrr_data.weight_1[i*35+18]*diffuseColor.z+rrrlrr_data.weight_1[i*35+19]*diffuseColor.w+rrrlrr_data.weight_1[i*35+20]*diffuse.x+rrrlrr_data.weight_1[i*35+21]*diffuse.y+rrrlrr_data.weight_1[i*35+22]*diffuse.z+rrrlrr_data.weight_1[i*35+23]*diffuse.w+rrrlrr_data.weight_1[i*35+24]*specular.x+rrrlrr_data.weight_1[i*35+25]*specular.y+rrrlrr_data.weight_1[i*35+26]*specular.z+rrrlrr_data.weight_1[i*35+27]*specular.w+rrrlrr_data.weight_1[i*35+28]*p3d_Material.emission.xyz.x+rrrlrr_data.weight_1[i*35+29]*p3d_Material.emission.xyz.y+rrrlrr_data.weight_1[i*35+30]*p3d_Material.emission.xyz.z+rrrlrr_data.weight_1[i*35+31]*isWater.x+rrrlrr_data.weight_1[i*35+32]*isWater.y+rrrlrr_data.weight_1[i*35+33]*isParticle.x+rrrlrr_data.weight_1[i*35+34]*isParticle.y;
    imt[i] = relu(imt[i]);
  }
  for (int i = 0; i < 8; i++) {
    oup[i] = rrrlrr_data.bias_2[i];
    for (int j = 0; j < 512; j++) {
      oup[i] += rrrlrr_data.weight_2[i*512+j] * imt[j];
    }
  }

        }
      }
    } else{
      float impossible = 1 / 0;
    }
  }

  out0.x = mclamp(oup[0]);
  out0.y = mclamp(oup[1]);
  out0.z = mclamp(oup[2]);
  out0.w = mclamp(oup[3]);

  out1.x = mclamp(oup[4]);
  out1.y = mclamp(oup[5]);
  out1.z = mclamp(oup[6]);
  out1.w = mclamp(oup[7]);

}
