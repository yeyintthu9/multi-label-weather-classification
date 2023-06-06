sudo add-apt-repository ppa:rmescandon/yq
sudo apt-get update -y && \
# install pyenv dependencies 
sudo apt-get install make build-essential libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
        libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
        yq -y # install other dependencies
# install pyenv
curl https://pyenv.run | bash
# set command
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# install pip
sudo apt-get install python3-pip
# install virtualenv and mlflow with pip
pip install virtualenv mlflow

echo 'Restarting shell..'
exec "$SHELL"

