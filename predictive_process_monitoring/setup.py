from setuptools import setup, find_packages

setup(name='predictive_process_monitoring',
      packages=find_packages(),
      package_data={
            "predictive_process_monitoring": ["VP/*.py", "prbpm_models/*.py", "prbpm_models/stored_models/traxens_ensemble/*.h5"],
      },
      include_package_data=True,
      version='1.0.1')
