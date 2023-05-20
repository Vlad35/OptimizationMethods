import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        double rk = 1;
        double c = 2;
        int N = 1000;
        double[] xk = {1.5, 1.5};
        double eps1 = Math.pow(10, -5);
        double eps2 = Math.pow(10, -5);

        while (true) {
            double f1 = F1(xk, rk);
            double[] x_0 = mns(xk, rk, eps1, eps2, N);
            double p1 = P1(x_0, rk);
            if (p1 < eps1) {
                double[] x_res = x_0;
                double f_res = F(x_0);
                System.out.println("Method of Penalty Functions");
                System.out.println("x = " + Arrays.toString(x_res));
                System.out.println("f(x) = " + f_res);
                break;
            } else {
                rk = c * rk;
                xk = x_0;
            }
        }
    }

    public static double F(double[] x) {
        double f = 2 * x[0] * x[0] + x[1] * x[1] - x[0] * x[1] + x[0];
        return f;
    }

    public static double g(double[] x) {
        double constraint = x[0] + x[1] - 3;
        return constraint;
    }

    public static double g1(double[] x) {
        return 0;
    }

    public static double P1(double[] x_0, double rk) {
        double constraint = g(x_0);
        double constraint1 = g1(x_0);
        double p1 = rk / 2 * (constraint * constraint + constraint1 * constraint1);
        return p1;
    }

    public static double F1(double[] x, double rk) {
        double objective = F(x);
        double constraint = g(x);
        double constraint1 = g1(x);
        double f1 = objective + rk / 2 * (constraint * constraint + constraint1 * constraint1);
        return f1;
    }

    public static double[] dF(double[] x, double rk) {
        double h = Math.pow(10, -6);
        double[] derivatives = new double[x.length];

        for (int i = 0; i < x.length; i++) {
            double[] x1 = Arrays.copyOf(x, x.length);
            x1[i] += h;
            derivatives[i] = (F1(x1, rk) - F1(x, rk)) / h;
        }

        return derivatives;
    }

    public static double[] sumV(double[] v1, double[] v2) {
        double[] sum = new double[v1.length];
        for (int i = 0; i < v1.length; i++) {
            sum[i] = v1[i] + v2[i];
        }
        return sum;
    }

    public static double[] diffV(double[] v1, double[] v2) {
        double[] diff = new double[v1.length];
        for (int i = 0; i < v1.length; i++) {
            diff[i] = v1[i] - v2[i];
        }
        return diff;
    }

    public static double[] v_multiply_k(double[] v1, double k) {
        double[] result = new double[v1.length];
        for (int i = 0; i < v1.length; i++) {
            result[i] = v1[i] * k;
        }
        return result;
    }

    public static double[] mns(double[] x0, double rk, double eps1, double eps2, int N) {
        double[] x_result = new double[x0.length];
        if (norma(dF(x0, rk)) < eps1) {
            x_result = x0;
        } else {
            int it = 0;
            int count = 0;
            while (true) {
                double[] s = v_multiply_k(dF(x0, rk), -1);
                double alpha = pd(x0, rk);
                double[] x = sumV(x0, v_multiply_k(s, alpha));
                if (norma(diffV(x, x0)) < eps2 && Math.abs(F1(x, rk) - F1(x0, rk)) < eps2) {
                    count++;
                    x_result = x;
                    if (count == 2) {
                        break;
                    }
                }
                x0 = x;
                if (it >= N) {
                    x_result = x;
                    break;
                }
                it++;
            }
        }
        return x_result;
    }

    public static double norma(double[] mas) {
        double max = Math.abs(mas[0]);
        for (int i = 1; i < mas.length; i++) {
            double absValue = Math.abs(mas[i]);
            if (absValue > max) {
                max = absValue;
            }
        }
        return max;
    }

    public static double pd(double[] x0, double rk) {
        double a = -100;
        double b = 100;
        double eps = Math.pow(10, -6);
        double delta = eps / 2;
        double[] df = dF(x0, rk);

        while (true) {
            double x1 = (a + b - delta) / 2;
            double[] newargX = new double[x0.length];
            for (int i = 0; i < x0.length; i++) {
                newargX[i] = x0[i] - x1 * df[i];
            }

            double y1 = (a + b + delta) / 2;
            double[] newargY = new double[x0.length];
            for (int i = 0; i < x0.length; i++) {
                newargY[i] = x0[i] - y1 * df[i];
            }

            if (F1(newargX, rk) < F1(newargY, rk)) {
                b = y1;
            } else {
                a = x1;
            }

            if (Math.abs(b - a) <= 2 * eps) {
                x1 = (a + b) / 2;
                return x1;
            }
        }
    }
}
