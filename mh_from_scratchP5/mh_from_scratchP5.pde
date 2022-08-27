//
//

float normal(float mean, float std){
  return randomGaussian()*std + mean;
}

float[] normal(float mean, float std, int size){
  float[] retval = zeros(size);
  for (int i = 0; i < size; ++i) {
    retval[i] = normal(mean, std);
  }
  return retval;
}

float[] transition_model_a(float[] x) {
  float[] retval = zeros(2);
  retval[0] = x[0];
  retval[1] = abs(normal(x[1], 0.5));
  return retval;
}

float prior_a(float[] x){
  if(x[1] <= 0) return 0.;
  return 1.;
}

boolean acceptance_a(float x, float x_new){
  if(x_new > x) return true;
  float accept = random(0, 1);
  return (accept < exp(x_new-x));
}

float log_lik_normal_a(float[] x, float[] data){
  float a = -log(x[1] * sqrt(2*PI));
  //float a2 = x[1] * sqrt(2*PI);
  println("a: " + a);
  float[] b = divide(pow(subtract(data, x[0]), 2), 2*sq(x[1]));
  printArray("b", b);
  float[] c = subtract(a, b);
  printArray("c", c);
  //float d2 = sumArray(multiply(-1, log(c)));
  float d = sumArray(c);
  println("d: " + d);
  return d;
}

int num = 100;
float std = 75;
float[] xs = normal(0, std, num);
float[] ys = normal(0, std, num);
float[] population = normal(10, 3, 30000);
int[] ixes = randomIntArray(1000, 30000);

float[] observation = take(population, ixes);

interface MHInterface{
  float log_likelihood(float[] x, float[] data);
  float[] transition_model(float[] x);
  boolean acceptance(float x, float x_new);
  float prior(float [] x);
}

class NormalMH implements MHInterface {
  

  float log_likelihood(float[] x, float[] data){
    return log_lik_normal_a(x, data);
  }
  float[] transition_model(float[] x){
    return transition_model_a(x);
  }
  boolean acceptance(float x, float x_new){
    return acceptance_a(x, x_new);
  }
  float prior(float [] x){
    return prior_a(x);
  }
}

float[] metropolis_hastings_single(MHInterface c, 
  float[] param_init, 
  float[] data) {

    float[] x = param_init;
    float[] ret = zeros(2);
    float[] x_new = c.transition_model(x);
    float x_lik = c.log_likelihood(x, data);
    float x_new_lik = c.log_likelihood(x_new, data);
    if(c.acceptance(x_lik + log(c.prior(x)),
      x_new_lik + log(prior(c.x_new))))
      ret[0] = 1; // accepted
    else 
      ret[0] = 0; // not accept
    ret[1] = x_new;
    return ret;
}

void setup(){
  size(400, 400);
  frameRate(1);
  float[] x = {10,2};
  
}

void draw(){
  
  background(51);
  fill(200);
  pushMatrix();
  translate(width/2, height/2);
  for (int i = 0; i < num; ++i) {
    circle(xs[i], ys[i], 10);  
  }
  popMatrix();
  
  
}
