# Split Bregman for TV Image Recovery

### Author: Anqi Ni
### Email: anqini4@gmail.com

## dependency:
- numpy
- scipy / imageio
- matplotlib

## tips:
1. Please download all modules (i.e. numpy, etc.) before you run the project.
2. `gaussian_method2` is an alternative of `gaussian_method` but it runs slower.
3. In `imnoise` function, it parameter **3** can be changed to other number, but **lambda** and **mu** must be changed accordingly to make sure it converge.

The following parameters workd pretty well:

| noise | lambd | mu   |
|-------|-------|------|
| 3     | 0.1   | 0.05 |
| 2     | 0.1   | 0.1  |
| 1     | 0.05  | 0.1  |

*you can change paramters as you want, the parameters above is just for reference.

Good Luck!

