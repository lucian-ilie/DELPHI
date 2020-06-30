#!/bin/bash
#set -x
mkdir -p $TMP_DIR



# ####################
# ##      ECO       ##
# #################### (verified)
echo "computing ECO"
${PRO_DIR}/feature_computation/ECO/run_ECO.sh 

# ####################
# ##      RSA       ##
# #################### (verified)
echo "computing RSA"
${PRO_DIR}/feature_computation/RSA/run_RSA.sh &

# ####################
# ##      RAA       ##
# #################### (verified)
echo "computing RAA"
${PRO_DIR}/feature_computation/RAA/run_RAA.sh &

# ####################
# ##      HYD       ##
# #################### (verified)
echo "computing HYD"
${PRO_DIR}/feature_computation/HYD/run_HYD.sh &

# ####################
# ##      PKA       ##
# #################### (verified)
echo "computing PKA"
${PRO_DIR}/feature_computation/PKA/run_PKA.sh &

# ####################
# ##   POSITION     ##
# #################### (verified)
echo "computing POSITION"
${PRO_DIR}/feature_computation/POSITION/run_POSITION.sh &

# ####################
# ##   PHY_Char     ##
# #################### (verified)
echo "computing PHY_Char"
${PRO_DIR}/feature_computation/PHY_Char/run_PHY_Char.sh &

# ####################
# ##   PHY_Prop     ##
# #################### (verified)
echo "computing PHY_Prop"
${PRO_DIR}/feature_computation/PHY_Prop/run_PHY_Prop.sh &

# ####################
# ##   Pro2Vec_1D   ##
# #################### (verified)
echo "computing Pro2Vec_1D"
${PRO_DIR}/feature_computation/Pro2Vec_1D/run_Pro2Vec_1D.sh &

# ####################
# ##      Anchor    ##
# #################### (verified)
echo "computing Anchor"
${PRO_DIR}/feature_computation/Anchor/run_Anchor.sh &

# ####################
# ##      HSP       ##
# #################### (verified)
echo "computing HSP"
${PRO_DIR}/feature_computation/HSP/run_HSP.sh 


####################
##      PSSM      ##
####################  
echo "computing PSSM" 
${PRO_DIR}/feature_computation/PSSM/run_PSSM.sh 