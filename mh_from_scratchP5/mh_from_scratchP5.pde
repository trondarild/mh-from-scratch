//
//
Buffer accept_buf = new Buffer(200);
Buffer reject_buf = new Buffer(200);

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
  float accept = random(1);
  return (accept < exp(x_new-x));
}

float log_lik_normal_a(float[] x, float[] data){
  float a = -log(x[1] * sqrt(2*PI));
  //float a = x[1] * sqrt(2*PI);
  float[] b = divide(sq(subtract(data, x[0])), 2*sq(x[1]));
  float[] c = subtract(a, b);
  //float d2 = sumArray(multiply(-1, log(c)));
  // printArray("c", c);
  // printArray("log abs c", log(abs(c)));
  float d = sumArray(multiply(-1, log(abs(c))));
  return d;
}

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

float[] histogram(float val, float[] hist, float min, float max){
  int ix = int(float(hist.length) * (val-min)/(max-min));
  if(ix >= 0 && ix < hist.length)
    hist[ix]+=1;
  return hist;

}

//int num = 100;
//float[] xs = normal(0, std, num);
//float[] ys = normal(0, std, num);
float std = 15;
float[] population = normal(0, std, 30000);
int[] ixes = randomIntArray(1000, 30000);
float[] observation = take(population, ixes);
float mu_obs = mean(observation);

NormalMH comp = new NormalMH();
float[] x = {mu_obs, 0.1};
float[] hist = zeros(11);

float[] metropolis_hastings_single(MHInterface c, 
  float[] param_init, 
  float[] data) {

    float[] x = param_init;
    float[] ret = zeros(2);
    float[] x_new = c.transition_model(x);
    float x_lik = c.log_likelihood(x, data);
    float x_new_lik = c.log_likelihood(x_new, data);
    if(c.acceptance(x_lik + log(c.prior(x)),
      x_new_lik + log(c.prior(x_new))))
      ret[0] = 1; // accepted
    else 
      ret[0] = 0; // not accept
    ret[1] = x_new[1];
    return ret;
}

void setup(){
  size(400, 400);
  frameRate(30);
  float[] tstval = {5, 0.1};
  //float tst = log_lik_normal_a(tstval, observation);
  //println("lln: " + tst);
  //float[] tst = transition_model_a(tstval);
  //println("lln: " + tst[0] + "; " + tst[1]);

}

void draw(){
  
  background(51);
  fill(200);
  
  float[] mh = metropolis_hastings_single(
    comp, x, observation
  );
  if(mh[0]==0) reject_buf.append(mh[1]);
  else accept_buf.append(mh[1]);

  
  pushMatrix();
  float[][] data = { reject_buf.array(), accept_buf.array()};
  translate(10,100);
  scale(1.5);
  drawTimeSeries(data, 0., 15., 0.5, 3.0, null);
  
  popMatrix();
  //println("accept: " + mh[0] + "; val: " + mh[1]);
  if(mh[0]==1) x = mh; // only update if accept
  
  pushMatrix();
  translate(30, 50);
  pushStyle();
  textSize(50);
  text("" + mean(accept_buf.array()), 0, 0); 
  text("" + accept_buf.tail(), 0, 55); 
  popStyle();
  popMatrix();
  float w = 4;
  hist = histogram(accept_buf.tail(), hist, std-w, std+w);
  //printArray("hist", hist);
  
  pushMatrix();
  textSize(10);
  translate(2*width/4, 3*height/4);
  scale(1.2);

  barchart_array(normalize(hist), "hist");

  popMatrix();
  
  
}
